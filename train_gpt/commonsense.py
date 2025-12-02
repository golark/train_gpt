import torch
import torch.nn.functional as F
import tiktoken
from tqdm import tqdm
from datasets import load_dataset


def evaluate_commonsense(model, split='validation', num_examples=None, device=None):
    """
    Evaluate a GPT-style model on CommonsenseQA (tau/commonsense_qa).
    """

    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")

    # Determine device
    if device is None:
        device = next(model.parameters()).device

    # Load dataset (train/validation/test)
    ds = load_dataset("tau/commonsense_qa")[split]

    # Optionally trim
    if num_examples is not None:
        ds = ds.select(range(num_examples))

    print(f"Loaded {len(ds)} examples from CommonsenseQA ({split} split)")

    # Extract answer key mapping (A,B,C,D,E -> 0..4)
    letter_to_index = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

    # Evaluation loop
    correct_count = 0
    num_total = len(ds)

    model.eval()
    with torch.no_grad():
        for example in tqdm(ds, desc="Evaluating"):

            question = example["question"]
            choices = example["choices"]["text"]
            label_letter = example["answerKey"]
            gold = letter_to_index[label_letter]

            # Tokenize prefix
            ctx = question.strip()
            ctx_tokens = enc.encode(ctx)

            # Tokenize all choices with a leading space
            endings_tokens = [
                enc.encode(" " + choice) for choice in choices
            ]

            option_logprobs = []

            # Evaluate each choice
            for ending_tokens in endings_tokens:
                full = ctx_tokens + ending_tokens
                x = torch.tensor(full, device=device).unsqueeze(0)

                logits, _ = model(x, targets=None)
                logprobs = F.log_softmax(logits, dim=-1)  # (1, T, vocab)

                # Compute sum of token logprobs for the ending
                token_lp = []
                for j, tok in enumerate(ending_tokens):
                    step = len(ctx_tokens) - 1 + j
                    token_lp.append(logprobs[0, step, tok].item())

                option_logprobs.append(sum(token_lp))

            # Predict best-scoring ending
            pred = max(range(len(choices)), key=lambda i: option_logprobs[i])

            if pred == gold:
                correct_count += 1

    accuracy = 100 * correct_count / num_total

    print("\nCommonsenseQA Evaluation Results:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Correct: {correct_count}/{num_total}")

    return {
        "accuracy": accuracy,
        "num_correct": correct_count,
        "num_total": num_total,
        "num_evaluated": num_total,
    }
