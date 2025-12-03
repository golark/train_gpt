Surpass GPT2 baseline model performance with a budget of $10 ( as of Late 2025 )


## Comparison


### how to make
| name                        | command                           | cost     |
| :-------------------------- | :-------------------------------- | :------- |
| GPT2 124M                   | `make baseline`                   |          |
| + Increased Max LR          | `make train_increase_max_lr`      |          |
| + Rotational Pos Embeddings | `make train_rotational_pos_emb`   |          |
| + SWigLU                    | [ ]                               |          |


### next
- [x] commonsense_qa
- [x] exp 1 - baseline gpt 
- [x] exp 2 - larger lr gpt2 until meet gpt2 perf
- [x] exp 3 - ROPE until meet gpt2 perf
- [x] gpt2 on commonsense_qa
- [ ] scale up the model
- [ ] exp 4 - Swiglu
- [ ] v3 model performance on commonsense_qa 