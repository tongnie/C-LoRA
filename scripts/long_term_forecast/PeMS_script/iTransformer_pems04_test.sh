export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

python -u run.py \
--task_name long_term_forecast \
--is_training 0 \
--root_path ./dataset/pems/ \
--data_path pems04.csv \
--model_id PEMS04_96_96  \
--model iTransformer \
--data custom \
--features M \
--seq_len 96 \
--label_len 48 \
--pred_len 96 \
--e_layers 3 \
--d_layers 1 \
--factor 3 \
--enc_in 307 \
--des 'Exp' \
--d_model 256 \
--d_ff 512 \
--learning_rate 0.001 \
--itr 1 \
--rank 32 \
--node_dim 64
