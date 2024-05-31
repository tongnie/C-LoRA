model_name=Uniformer

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id weather_96_96 \
    --model Uniformer \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --e_layers 3 \
    --n_heads 1 \
    --enc_in 21 \
    --des 'Exp' \
    --input_dim 1 \
    --node_dim 20 \
    --embed_dim 512 \
    --d_ff 512  \
    --itr 1 \
    --if_node \
    --num_layer 4


python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id weather_96_192 \
    --model Uniformer \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 192 \
    --e_layers 3 \
    --n_heads 1 \
    --enc_in 21 \
    --des 'Exp' \
    --input_dim 1 \
    --node_dim 20 \
    --embed_dim 512 \
    --d_ff 512  \
    --itr 1 \
    --if_node \
    --num_layer 4


python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id weather_96_336 \
    --model Uniformer \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 336 \
    --e_layers 3 \
    --n_heads 1 \
    --enc_in 21 \
    --des 'Exp' \
    --input_dim 1 \
    --node_dim 20 \
    --embed_dim 512 \
    --d_ff 512 \
    --itr 1 \
    --if_node \
    --num_layer 4


python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id weather_96_720 \
    --model Uniformer \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 720 \
    --e_layers 2 \
    --n_heads 1 \
    --enc_in 21 \
    --des 'Exp' \
    --input_dim 1 \
    --node_dim 20 \
    --embed_dim 512 \
    --d_ff 512 \
    --itr 1 \
    --if_node \
    --num_layer 4

