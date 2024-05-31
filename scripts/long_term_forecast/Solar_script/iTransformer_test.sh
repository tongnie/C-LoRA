export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

python -u run.py \
    --task_name long_term_forecast \
    --is_training 0 \
    --root_path ./dataset/solar/ \
    --data_path solar.csv \
    --model_id SOLAR_96_96 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --e_layers 3 \
    --d_layers 1 \
    --enc_in 137 \
    --dec_in 137 \
    --c_out 137 \
    --des 'Exp' \
    --d_model 256 \
    --d_ff 512 \
    --learning_rate 0.0005 \
    --itr 1 \
    --rank 32 \
    --node_dim 64 \
