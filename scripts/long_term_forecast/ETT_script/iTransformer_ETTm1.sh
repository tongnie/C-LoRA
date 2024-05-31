export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1  \
    --root_path ./dataset/ETT-small/  \
    --data_path ETTm1.csv \
    --model_id ETTm1_96_96 \
    --model iTransformer \
    --data ETTm1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 128  \
    --d_ff 128 \
    --itr 1 \
    --rank 32 \
    --node_dim 64

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1  \
    --root_path ./dataset/ETT-small/  \
    --data_path ETTm1.csv \
    --model_id ETTm1_96_192 \
    --model iTransformer \
    --data ETTm1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 192 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 128 \
    --d_ff 128 \
    --itr 1 \
    --rank 32 \
    --node_dim 64

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1  \
    --root_path ./dataset/ETT-small/  \
    --data_path ETTm1.csv \
    --model_id ETTm1_96_336 \
    --model iTransformer \
    --data ETTm1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 336 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 128 \
    --d_ff 128 \
    --itr 1 \
    --rank 32 \
    --node_dim 64

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1  \
    --root_path ./dataset/ETT-small/  \
    --data_path ETTm1.csv \
    --model_id ETTm1_96_720 \
    --model iTransformer \
    --data ETTm1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 720 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 128 \
    --d_ff 128 \
    --itr 1 \
    --rank 32 \
    --node_dim 64

