

model_name=Uniformer

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/pems/ \
    --data_path pems04.csv \
    --model_id PEMS04_96_96 \
    --model Uniformer \
    --data custom \
    --features M \
    --seq_len 96  \
    --pred_len 96 \
    --e_layers 3 \
    --n_heads 1 \
    --enc_in 307 \
    --input_dim 1  \
    --rank_factor 10 \
    --node_dim 64 \
    --embed_dim 256 \
    --d_ff 512  \
    --num_layer 1 \
    --des 'Exp' \
    --itr 1   \
    --learning_rate 0.001 \
    --train_epochs 15 \
    --if_node
    #--node_dim 128

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/pems/ \
    --data_path pems04.csv \
    --model_id PEMS04_96_192 \
    --model Uniformer \
    --data custom \
    --features M \
    --seq_len 96  \
    --pred_len 192 \
    --e_layers 3 \
    --n_heads 1 \
    --enc_in 307 \
    --input_dim 1  \
    --rank_factor 10 \
    --node_dim 64 \
    --embed_dim 256 \
    --d_ff 512  \
    --num_layer 1 \
    --des 'Exp' \
    --itr 1   \
    --learning_rate 0.001 \
    --train_epochs 15 \
    --if_node
    #--node_dim 128

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/pems/ \
    --data_path pems04.csv \
    --model_id PEMS04_96_326 \
    --model Uniformer \
    --data custom \
    --features M \
    --seq_len 96  \
    --pred_len 336 \
    --e_layers 3 \
    --n_heads 1 \
    --enc_in 307 \
    --input_dim 1  \
    --rank_factor 10 \
    --node_dim 64 \
    --embed_dim 256 \
    --d_ff 512  \
    --num_layer 1 \
    --des 'Exp' \
    --itr 1   \
    --learning_rate 0.001 \
    --train_epochs 15 \
    --if_node
    #--node_dim 128

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/pems/ \
    --data_path pems04.csv \
    --model_id PEMS04_96_720 \
    --model Uniformer \
    --data custom \
    --features M \
    --seq_len 96  \
    --pred_len 720 \
    --e_layers 3 \
    --n_heads 1 \
    --enc_in 307 \
    --input_dim 1  \
    --rank_factor 10 \
    --node_dim 64 \
    --embed_dim 256 \
    --d_ff 512  \
    --num_layer 1 \
    --des 'Exp' \
    --itr 1   \
    --learning_rate 0.001 \
    --train_epochs 15 \
    --if_node
    #--node_dim 128
