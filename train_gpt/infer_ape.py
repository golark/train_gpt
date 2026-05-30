import argparse
import sys
import types
from pathlib import Path

import tiktoken
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import model_ape as _ma
from model_ape import GPT, GPTConfig

# checkpoint may reference train_gpt.model.GPTConfig
_shim = types.ModuleType("train_gpt")
_shim.model = _ma
sys.modules.setdefault("train_gpt", _shim)
sys.modules.setdefault("train_gpt.model", _ma)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def load_model(checkpoint_path: Path) -> GPT:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        config = checkpoint.get("config", GPTConfig())
    else:
        state_dict = checkpoint
        config = GPTConfig()

    state_dict = {
        k.replace("_orig_mod.", "").replace("module.", ""): v
        for k, v in state_dict.items()
    }

    model = GPT(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def generate(model: GPT, enc, prompt: str, max_new_tokens: int) -> str:
    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(x)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            x = torch.cat([x, next_token], dim=1)

    return enc.decode(x[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT.parent / "models" / "inc_max_lr_model.pt",
    )
    parser.add_argument(
        "--prompt",
        default="Hello, I'm a language model,",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument(
        "--save-safetensors",
        type=Path,
        default=None,
        help="optional path to export weights as safetensors",
    )
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    config = model.config
    print(
        f"Loaded APE model on {device} | "
        f"layers={config.n_layer} heads={config.n_head} embd={config.n_embd} "
        f"vocab={config.vocab_size}"
    )

    enc = tiktoken.get_encoding("gpt2")
    output = generate(model, enc, args.prompt, args.max_new_tokens)

    print(f"\nPrompt: {args.prompt!r}")
    print(f"Output: {output}")

    if args.save_safetensors is not None:
        from safetensors.torch import save_file

        save_file(
            {k: v.contiguous().cpu() for k, v in model.state_dict().items()},
            args.save_safetensors,
        )
        print(f"Saved safetensors to {args.save_safetensors}")


if __name__ == "__main__":
    main()
