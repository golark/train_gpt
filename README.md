- Repositry to Develop baseline models designed to surpass GPT2 performance, while keeping total training costs under $10 (estimated GPU rental prices for late 2025).
- Applies current state-of-the-art research and methods to reduce both training duration and expenses ( $ )
- Uses a GPT2-style architecture with 124M parameters as the reference point, and compares it against models enhanced with optimized learning rates, SwigLU activations, and by replacing absolute positional embeddings with rotary positional embeddings.


### Comparison
![](media/convergence_comparison.png)
![](media/commonsense_qa.png)


### train models
| name                        | command                         |
| :-------------------------- | :------------------------------ |
| GPT2 124M Baseline                  | `make baseline`                 |
| Increased Max LR          | `make train_increase_max_lr`    |
| Rotational Pos Embeddings | `make train_rotational_pos_emb` |


### next
- [x] commonsense_qa
- [x] exp 1 - baseline gpt 
- [x] exp 2 - larger lr gpt2 until meet gpt2 perf
- [x] exp 3 - ROPE until meet gpt2 perf
- [ ] exp 4 - Swiglu
- [x] gpt2 on commonsense_qa
- [ ] scale up the model
- [ ] v3 model performance on commonsense_qa 
- [ ] quantized inference and benchmark
- [ ] vanilla c serve model