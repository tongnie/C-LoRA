

model_name=Uniformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_96 \
  --model Uniformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 4 \
  --n_heads 1 \
  --enc_in 862 \
  --input_dim 1 \
  --node_dim 256 \
  --embed_dim 256 \
  --d_ff 512  \
  --num_layer 3 \
  --des 'Exp' \
  --batch_size 16 \
  --learning_rate 0.001 \
  --train_epochs 15 \
  --if_node
  #  --node_dim 256 \

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id traffic_96_192 \
#   --model Uniformer \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 192 \
#   --e_layers 3 \
#   --d_layers 1 \
#   --enc_in 862 \
#   --input_dim 1 \
#   --node_dim 128 \
#   --embed_dim 128 \
#   --d_ff 256 \
#   --num_layer 2 \
#   --des 'Exp' \
#   --batch_size 16 \
#   --learning_rate 0.001 \
#   --itr 1
#   --if_node

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id traffic_96_336 \
#   --model Uniformer \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 336 \
#   --e_layers 3 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 862 \
#   --input_dim 1 \
#   --node_dim 128 \
#   --embed_dim 128 \
#   --d_ff 256  \
#   --num_layer 2 \
#   --des 'Exp' \
#   --batch_size 16 \
#   --learning_rate 0.001 \
#   --itr 1
#   --if_node

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id traffic_96_720 \
#   --model Uniformer \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 720 \
#   --e_layers 3 \
#   --n_heads 1 \
#   --enc_in 862 \
#   --input_dim 1 \
#   --node_dim 128 \
#   --embed_dim 128 \
#   --d_ff 256  \
#   --num_layer 2 \
#   --des 'Exp' \
#   --itr 1 \
#   --train_epochs 15 \
#   --if_node
