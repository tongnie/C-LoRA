model_name=Uniformer



python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ECL_96_96  \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96  \
    --pred_len 96 \
    --e_layers 3 \
    --n_heads 1 \
    --enc_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --node_dim 128 \
    --input_dim 1 \
    --embed_dim 256 \
    --d_ff 512 \
    --num_layer 1 \
    --batch_size 16 \
    --learning_rate 0.0005 \
    --itr 1 \
    --if_node \


python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ECL_96_192  \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96  \
    --pred_len 192 \
    --e_layers 3 \
    --n_heads 1 \
    --enc_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --node_dim 128 \
    --input_dim 1 \
    --embed_dim 256  \
    --d_ff 512 \
    --num_layer 1 \
    --batch_size 16 \
    --learning_rate 0.0005\
    --itr 1 \
    --if_node \


python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ECL_96_336  \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96  \
    --pred_len 336 \
    --e_layers 3 \
    --n_heads 1 \
    --enc_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --node_dim 128 \
    --input_dim 1 \
    --embed_dim 256  \
    --d_ff 512 \
    --num_layer 1 \
    --batch_size 16 \
    --learning_rate 0.0005 \
    --itr 1 \
    --if_node \


python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ECL_96_720  \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96  \
    --pred_len 720 \
    --e_layers 3 \
    --n_heads 1  \
    --enc_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --node_dim 128 \
    --input_dim 1 \
    --embed_dim 256  \
    --d_ff 512 \
    --num_layer 1 \
    --batch_size 16 \
    --learning_rate 0.0005 \
    --itr 1 \
    --if_node \

