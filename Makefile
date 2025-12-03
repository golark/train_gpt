
train_increase_max_lr:
	python3 -m train_gpt.train \
		--vocab-size 50304 \
		--block-size 1024 \
		--n-layer 12 \
		--n-head 12 \
		--n-embd 768 \
		--batch-size 64 \
		--seq-len 1024 \
		--total-batch-size 524288 \
		--epoch-idx 0 \
		--max-steps 19073 \
		--max-lr 1.5e-3 \
		--min-lr 0.00015 \
		--warmup-steps 300 \
		--base-lr 6e-4 \
		--weight-decay 0.1 \
		--output-dir ./experiments/inc_max_lr

train_rotational_pos_emb: 
	python3 -m train_gpt.train \
	    --model model_rope \
		--vocab-size 50304 \
		--block-size 1024 \
		--n-layer 12 \
		--n-head 12 \
		--n-embd 768 \
		--batch-size 64 \
		--seq-len 1024 \
		--total-batch-size 524288 \
		--epoch-idx 0 \
		--max-steps 19073 \
		--max-lr 1.5e-3 \
		--min-lr 0.00015 \
		--warmup-steps 300 \
		--base-lr 6e-4 \
		--weight-decay 0.1 \
		--output-dir ./experiments/rope

fastest: train_rotational_pos_emb