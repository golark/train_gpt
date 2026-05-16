import sys
import torch
import tiktoken

sys.path.insert(0, '/Users/cjan/golark/train_gpt/train_gpt')
from model_ape import GPT, GPTConfig

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# checkpoint was saved with train_gpt.model.GPTConfig; shim that path to model_ape
import types, model_ape as _ma
_shim = types.ModuleType('train_gpt')
_shim.model = _ma
sys.modules.setdefault('train_gpt', _shim)
sys.modules.setdefault('train_gpt.model', _ma)

checkpoint = torch.load('/Users/cjan/golark/train_gpt/models/inc_max_lr_model.pt', map_location=device, weights_only=False)

if isinstance(checkpoint, dict) and 'model' in checkpoint:
    state_dict = checkpoint['model']
    config = checkpoint.get('config', GPTConfig())
else:
    state_dict = checkpoint
    config = GPTConfig()

state_dict = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in state_dict.items()}

model = GPT(config)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

print(f"Loaded APE model on {device} | layers={config.n_layer} heads={config.n_head} embd={config.n_embd}")

enc = tiktoken.get_encoding('gpt2')

prompts = [
    "Hello, I'm a language model,",
    "The capital of France is",
    "Once upon a time,",
]

num_return_sequences = 1
max_new_tokens = 50

for prompt in prompts:
    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0).repeat(num_return_sequences, 1)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(x)
            next_logits = logits[:, -1, :]
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_token], dim=1)

    for i in range(num_return_sequences):
        generated = enc.decode(x[i].tolist())
        print(f"\nPrompt: {prompt!r}")
        print(f"Output: {generated}")
