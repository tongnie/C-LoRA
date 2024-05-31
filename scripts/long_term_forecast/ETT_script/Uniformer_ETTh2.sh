

model_name=Uniformer


python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id ETTh2_96_96 \
    --model Uniformer \
    --data ETTh2 \
    --features M \
    --seq_len 96  \
    --pred_len 96 \
    --e_layers 3  \
    --enc_in 7 \
    --input_dim 1 \
    --node_dim 0 \
    --embed_dim 512 \
    --d_ff 512  \
    --num_layer 1 \
    --des 'Exp' \
    --itr 1 \
    --if_node

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id ETTh2_96_192 \
    --model Uniformer \
    --data ETTh2 \
    --features M \
    --seq_len 96  \
    --pred_len 192 \
    --e_layers 3  \
    --enc_in 7 \
    --input_dim 1 \
    --node_dim 0 \
    --embed_dim 512 \
    --d_ff 512  \
    --num_layer 1 \
    --des 'Exp' \
    --itr 1  \
    --if_node

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id ETTh2_96_336 \
    --model Uniformer \
    --data ETTh2 \
    --features M \
    --seq_len 96  \
    --pred_len 336 \
    --e_layers 3  \
    --enc_in 7 \
    --input_dim 1 \
    --node_dim 0 \
    --embed_dim 512 \
    --d_ff 512  \
    --num_layer 1 \
    --des 'Exp' \
    --itr 1  \
    --if_node

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id ETTh2_96_720 \
    --model Uniformer \
    --data ETTh2 \
    --features M \
    --seq_len 96 \
    --pred_len 720 \
    --e_layers 3  \
    --enc_in 7 \
    --input_dim 1 \
    --node_dim 0 \
    --embed_dim 512 \
    --d_ff 512  \
    --num_layer 2 \
    --des 'Exp' \
    --itr 1  \
    --if_node
