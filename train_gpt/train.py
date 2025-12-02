import torch
import torch.nn.functional as F
from collections import OrderedDict
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import tiktoken
import time
import os
import json
import math
import argparse

from .model import GPT, GPTConfig, sample_from_model
from .commonsense import evaluate_commonsense
from .dataloader import DataLoaderLite


def set_lr_with_warmup(step, optimizer, max_lr, min_lr, warmup_steps, max_steps):
    # 1) linear warmup for warmup_iters steps
    if step < warmup_steps:
        lr = max_lr * (step+1) / warmup_steps
    # 2) if > max_steps, return min learning rate
    elif step > max_steps:
        lr = min_lr
    else:
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        decay_ratio = min(max(decay_ratio, 0), 1)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        lr = min_lr + coeff * (max_lr - min_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_val_loss(data_loader, model, device):
    model.eval()
    data_loader.reset()
    device_type = torch.device(device).type

    val_loss_accum = 0.0
    val_loss_steps = 20

    with torch.no_grad():
        for _ in range(val_loss_steps):
            x, y = data_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.float16):
                _, loss = model(x, y)
            val_loss_accum += loss.detach().float().item() / val_loss_steps

    return val_loss_accum

def checkpoint(model, train_loss, val_loss, device, epoch, step, checkpoint_dir):
    commonsense = evaluate_commonsense(model, device=device)
    checkpoint = {
        'step': step,
        'val_loss': float(val_loss),
        'commonsense': commonsense['accuracy'],
        'train_loss': {str(k): float(v) for k, v in train_loss.items()}
    }

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoints_path = os.path.join(checkpoint_dir, "checkpoints.json")
    checkpoints = []
    if os.path.exists(checkpoints_path):
        with open(checkpoints_path, "r") as f:
            try:
                checkpoints = json.load(f)
            except Exception:
                checkpoints = []
    checkpoints.append(checkpoint)

    with open(checkpoints_path, "w") as f:
        json.dump(checkpoints, f, indent=2)

    print(f"Checkpoint {step} (epoch {epoch}) - val_loss: {val_loss:.6f}, Common Sense: {commonsense['accuracy']:.4f}")

def train_micro_steps(model, optimizer, num_micro_steps, train_loader, device, ddp_sync):

    model.train()
    optimizer.zero_grad()
    device_type = torch.device(device).type

    loss_accum = 0.0
    for step in range(num_micro_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if ddp_sync:
            model.require_backward_grad_sync = (step == num_micro_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss = loss / num_micro_steps
        loss_accum += loss.detach().float().item()
        loss.backward()

    if ddp_sync:
        # Only synchronize actual metrics/tensors, not python floats
        loss_tensor = torch.tensor(loss_accum, device=device, dtype=torch.float32)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        return loss_tensor.item() / num_micro_steps

    return loss_accum / num_micro_steps

def epoch(epoch_idx, model, train_loader, val_loader, device, optimizer, num_micro_steps, proc_idx, num_procs, max_steps, set_lr, output_dir):

    train_loss = OrderedDict()
    is_master = proc_idx == 0
    ddp_sync = num_procs > 1

    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # ------------------------ 
        # val loss and checkpoint |
        if step and (step % 100 == 0 or last_step):
            val_loss = get_val_loss(val_loader, model, device)
            checkpoint(model, train_loss, val_loss, device, epoch_idx, step, output_dir)
            train_loss = OrderedDict()
            print(f"[val] step {step} | val_loss: {val_loss:.6f}")
            os.makedirs(output_dir, exist_ok=True)
            torch.save({'model': model.state_dict(), 'epoch': epoch_idx, 'step':step, 'config': model.config }, f"{output_dir}/model.pt")

        # ----------------------
        # train                 |
        train_loss[step] = train_micro_steps(model, optimizer, num_micro_steps, train_loader, device, ddp_sync)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        set_lr(step, optimizer)
        optimizer.step()
        if "cuda" in device:
            torch.cuda.synchronize()

        if is_master:
            dt = time.time() - t0
            tokens_per_sec = ( train_loader.B * train_loader.T ) / dt
            print(f"step {step:5d} | loss: {list(train_loss.values())[-1]:.6f} | norm: {grad_norm:.4f} | "
                  f"dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train GPT model')
    
    # Model configuration
    parser.add_argument('--vocab-size', type=int, default=50304, help='Vocabulary size')
    parser.add_argument('--block-size', type=int, default=1024, help='Max sequence length (block size)')
    parser.add_argument('--n-layer', type=int, default=12, help='Number of transformer layers')
    parser.add_argument('--n-head', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--n-embd', type=int, default=768, help='Embedding dimension')
    
    # Training configuration
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (B)')
    parser.add_argument('--seq-len', type=int, default=1024, help='Sequence length (T)')
    parser.add_argument('--total-batch-size', type=int, default=524288, help='Total batch size in tokens (2^19 ~0.5M)')
    parser.add_argument('--epoch-idx', type=int, default=0, help='Starting epoch index')
    parser.add_argument('--max-steps', type=int, default=19073, help='Maximum training steps (~1 epoch)')
    
    # Learning rate configuration
    parser.add_argument('--max-lr', type=float, default=1.5e-3, help='Maximum learning rate')
    parser.add_argument('--min-lr', type=float, default=None, help='Minimum learning rate (default: max_lr * 0.1)')
    parser.add_argument('--warmup-steps', type=int, default=300, help='Warmup steps')
    parser.add_argument('--base-lr', type=float, default=6e-4, help='Base learning rate for optimizer')
    
    parser.add_argument('--weight-decay', type=float, default=0.1, help='Weight decay')
    
    parser.add_argument('--output-dir', type=str, default='./checkpoints', help='Directory to save outputs')
    
    parser.add_argument('--device', type=str, default=None, help='Device to use (cpu/cuda/mps), auto-detected if not specified')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set minimum learning rate if not provided
    if args.min_lr is None:
        args.min_lr = args.max_lr * 0.1
    
    proc_rank = 0
    proc_local_rank = 0
    num_procs = 1

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    print(f"using device: {device}")

    if int(os.environ.get('RANK', -1)) != -1:
        assert torch.cuda.is_available(), "DDP requires CUDA presently."
        init_process_group(backend='nccl')
        proc_rank = int(os.environ['RANK'])
        proc_local_rank = int(os.environ['LOCAL_RANK'])
        num_procs = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{proc_local_rank}'
        torch.cuda.set_device(proc_local_rank)

    # Create data loaders
    B, T = args.batch_size, args.seq_len
    train_loader = DataLoaderLite(B, T, 'train', proc_rank, num_procs)
    val_loader = DataLoaderLite(B, T, 'val', proc_rank, num_procs)

    enc = tiktoken.get_encoding("gpt2")
    torch.set_float32_matmul_precision('high')

    device_type = torch.device(device).type

    # Calculate micro steps for gradient accumulation
    num_micro_steps = int(args.total_batch_size // (B * T))

    model_config = GPTConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd
    )
    model = GPT(model_config)
    model = torch.compile(model)
    model.to(device)

    if num_procs > 1:
        model = DDP(model, device_ids=[proc_local_rank])

    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.base_lr,
        device_type=device_type
    )
    from functools import partial
    set_lr = partial(
        set_lr_with_warmup,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train
    epoch(
        args.epoch_idx,
        model,
        train_loader,
        val_loader,
        device,
        optimizer,
        num_micro_steps,
        proc_rank,
        num_procs,
        args.max_steps,
        set_lr,
        args.output_dir
    )

if __name__ == '__main__':
    main()
