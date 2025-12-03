# 🚀 GPT2 Performance Sub-$10

Welcome to the **GPT2-Beaters** repo! Here we build and benchmark models intended to **outperform GPT2** (124M) — all while keeping total training costs **under $10** (estimated GPU rental, late 2025). 💸⚡️

- 🧠 **Cutting-edge Techniques:** Integrating the latest SOTA research to slash training time *and* expenses!
- 🏗️ **Reference Point:** Standard GPT2 (124M params) as baseline, compared with improved models using:
    - 🔥 Smarter learning rate schedules
    - 🍦 SwigLU activations
    - 🔄 Rotary positional embeddings replacing absolute pos. embeddings

---

## 📈 Model Performance & Comparison

| 🚦 Progress Over Baseline  |
|:-------------------------:|
| ![](media/convergence_comparison.png) |
| ![](media/commonsense_qa.png)         |

---

## 🛠️ Training Models Made Easy

| 📚 Model Variant           | ▶️ Command                      |
|:-------------------------- |:-------------------------------|
| 🟦 GPT2 124M Baseline      | `make baseline`                 |
| 🟥 Increased Max LR        | `make train_increase_max_lr`    |
| 🟩 Rotational Pos Emb      | `make train_rotational_pos_emb` |

---

## 🗺️ What’s Next?

- [x] ✅ commonsense_qa
- [x] ✅ exp 1 - baseline gpt
- [x] ✅ exp 2 - larger lr gpt2 until meet gpt2 perf
- [x] ✅ exp 3 - ROPE until meet gpt2 perf
- [ ] ✳️ exp 4 - Swiglu
- [x] ✅ gpt2 on commonsense_qa
- [ ] ⏫ scale up the model
- [ ] 📊 v3 model performance on commonsense_qa 
- [ ] 🪶 quantized inference and benchmark
- [ ] 🧑‍💻 vanilla C serve model

---

Let’s make *state-of-the-art* cheap, fun, and open! 🌟