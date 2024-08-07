export CUDA_VISIBLE_DEVICES=0

model_name=FreTS

# python -u run.py \
# --task_name long_term_forecast \
# --is_training 1 \
# --root_path ./dataset/pems/ \
# --data_path pems08.csv \
# --model_id PEMS08_96_96  \
# --model $model_name \
# --data custom \
# --features M \
# --seq_len 96 \
# --label_len 48 \
# --pred_len 96 \
# --e_layers 3 \
# --d_layers 1 \
# --enc_in 170  \
# --des 'Exp' \
# --d_model 256 \
# --d_ff 512 \
# --learning_rate 0.001 \
# --itr 1 \
# --rank 16 \
# --node_dim 32

# python -u run.py \
# --task_name long_term_forecast \
# --is_training 1 \
# --root_path ./dataset/pems/ \
# --data_path pems08.csv \
# --model_id PEMS08_96_192  \
# --model $model_name \
# --data custom \
# --features M \
# --seq_len 96 \
# --label_len 48 \
# --pred_len 192 \
# --e_layers 3 \
# --d_layers 1 \
# --enc_in 170  \
# --des 'Exp' \
# --d_model 256 \
# --d_ff 512 \
# --learning_rate 0.001 \
# --itr 1 \
# --rank 16 \
# --node_dim 32

python -u run.py \
--task_name long_term_forecast \
--is_training 1 \
--root_path ./dataset/pems/ \
--data_path pems08.csv \
--model_id PEMS08_96_336  \
--model $model_name \
--data custom \
--features M \
--seq_len 96 \
--label_len 48 \
--pred_len 336 \
--e_layers 3 \
--d_layers 1 \
--enc_in 170  \
--des 'Exp' \
--d_model 256 \
--d_ff 512 \
--learning_rate 0.001 \
--itr 1 \
--rank 16 \
--node_dim 32

# python -u run.py \
# --task_name long_term_forecast \
# --is_training 1 \
# --root_path ./dataset/pems/ \
# --data_path pems08.csv \
# --model_id PEMS08_96_720  \
# --model $model_name \
# --data custom \
# --features M \
# --seq_len 96 \
# --label_len 48 \
# --pred_len 720 \
# --e_layers 3 \
# --d_layers 1 \
# --enc_in 170  \
# --des 'Exp' \
# --d_model 256 \
# --d_ff 512 \
# --learning_rate 0.001 \
# --itr 1 \
# --rank 16 \
# --node_dim 32     
