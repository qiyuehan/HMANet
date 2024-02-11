

for pred_len in 96 192 336 720
do
  python -u LSTF_forecasting_96.py \
  --data ETTh1.csv \
  --model_name ETTh1_new \
  --pre_train 960 \
  --seq_len 96 \
  --pred_len $pred_len$ \
  --fea_dim 7 \
  --patch_size 16 \
  --n_hidden_dim 400 \
  --train_percent 0.5 \
  --epochs 20 \
  --batch_size 32 \
  --patience 5 \
  --add_nodes 5 \
  --test_percent 0.2


  python -u LSTF_forecasting_96.py \
  --data ETTh2.csv \
  --model_name ETTh2_new \
  --pre_train 960 \
  --seq_len 96 \
  --pred_len $pred_len$ \
  --fea_dim 7 \
  --patch_size 16 \
  --n_hidden_dim 1000 \
  --train_percent 0.5 \
  --epochs 20 \
  --batch_size 32 \
  --patience 5 \
  --add_nodes 5 \
  --test_percent 0.2

  python -u LSTF_forecasting_96.py \
  --data ETTm1.csv \
  --model_name ETTm1_new_0.7 \
  --pre_train 960 \
  --seq_len 96 \
  --pred_len $pred_len$ \
  --fea_dim 7 \
  --patch_size 16 \
  --n_hidden_dim 800 \
  --train_percent 0.7 \
  --epochs 20 \
  --batch_size 32 \
  --patience 5 \
  --add_nodes 5 \
  --test_percent 0.2

  python -u LSTF_forecasting_96.py \
  --data ETTm2.csv \
  --model_name ETTm2_new \
  --pre_train 960 \
  --seq_len 96 \
  --pred_len $pred_len$ \
  --fea_dim 7 \
  --patch_size 16 \
  --n_hidden_dim 1200 \
  --train_percent 0.7 \
  --epochs 20 \
  --batch_size 32 \
  --patience 5 \
  --add_nodes 5 \
  --test_percent 0.2



  python -u LSTF_forecasting_96.py \
  --data weater.csv \
  --model_name weather_new \
  --pre_train 960 \
  --seq_len 96 \
  --pred_len $pred_len$ \
  --fea_dim 21 \
  --patch_size 16 \
  --n_hidden_dim 1000 \
  --train_percent 0.7 \
  --epochs 20 \
  --batch_size 32 \
  --patience 5 \
  --add_nodes 5 \
  --test_percent 0.2


  python -u LSTF_forecasting_96.py \
  --data exchange.csv \
  --model_name exchange_new \
  --pre_train 960 \
  --seq_len 96 \
  --pred_len $pred_len$ \
  --fea_dim 8 \
  --patch_size 16 \
  --n_hidden_dim 20 \
  --train_percent 0.7 \
  --epochs 10 \
  --batch_size 32 \
  --patience 3 \
  --add_nodes 5 \
  --test_percent 0.2

# 96(hidden=400),192,336=300  720=200
  python -u LSTF_forecasting_96.py \
  --data electricity.csv \
  --model_name ELC_new \
  --pre_train 960 \
  --seq_len 96 \
  --pred_len $pred_len$ \
  --fea_dim 321 \
  --patch_size 16 \
  --n_hidden_dim 300 \
  --train_percent 0.7 \
  --epochs 20 \
  --batch_size 32 \
  --patience 3 \
  --add_nodes 5 \
  --test_percent 0.2 \
  --stride 4

# 96,192 (stride=2), 336, 720=(stride=4)
  python -u LSTF_forecasting_96.py \
  --data traffic.csv \
  --model_name traffic_new \
  --pre_train 960 \
  --seq_len 96 \
  --pred_len $pred_len$ \
  --fea_dim 862 \
  --patch_size 16 \
  --n_hidden_dim 300 \
  --train_percent 0.7 \
  --epochs 50 \
  --batch_size 32 \
  --patience 3 \
  --add_nodes 5 \
  --test_percent 0.2  \
  --stride 4


# 24-(n_hidden_dim 30; epoches:50; patience=5) other = n_hidden_dim=30, epoches=10, pathence=3)
  python -u LSTF_forecasting_96.py \
  --data ili.csv \
  --model_name ili_new \
  --pre_train 96 \
  --seq_len 60 \
  --pred_len  $pred_len$ \
  --fea_dim 7 \
  --patch_size 12 \
  --n_hidden_dim 30 \
  --train_percent 0.7 \
  --epochs 10 \
  --batch_size 32 \
  --patience 3 \
  --add_nodes 5 \
  --test_percent 0.2

done



  python -u LSTF_forecasting_96.py \
  --data exchange.csv \
  --model_name exchange_new \
  --pre_train 960 \
  --seq_len 96 \
  --pred_len 96 \
  --fea_dim 8 \
  --patch_size 16 \
  --n_hidden_dim 20 \
  --train_percent 0.7 \
  --epochs 50 \
  --batch_size 32 \
  --patience 3 \
  --add_nodes 5 \
  --test_percent 0.2  \
  --stride 1

   python -u LSTF_forecasting_96.py \
  --data exchange.csv \
  --model_name exchange_new \
  --pre_train 960 \
  --seq_len 96 \
  --pred_len 192 \
  --fea_dim 8 \
  --patch_size 16 \
  --n_hidden_dim 20 \
  --train_percent 0.7 \
  --epochs 5 \
  --batch_size 32 \
  --patience 3 \
  --add_nodes 5 \
  --test_percent 0.2  \
  --stride 1

   python -u LSTF_forecasting_96.py \
  --data exchange.csv \
  --model_name exchange_new \
  --pre_train 960 \
  --seq_len 96 \
  --pred_len 336 \
  --fea_dim 8 \
  --patch_size 16 \
  --n_hidden_dim 5 \
  --train_percent 0.7 \
  --epochs 1 \
  --batch_size 32 \
  --patience 3 \
  --add_nodes 5 \
  --test_percent 0.2  \
  --stride 1

   python -u LSTF_forecasting_96.py \
  --data exchange.csv \
  --model_name exchange_new \
  --pre_train 960 \
  --seq_len 96 \
  --pred_len 720 \
  --fea_dim 8 \
  --patch_size 16 \
  --n_hidden_dim 3 \
  --train_percent 0.7 \
  --epochs 1 \
  --batch_size 32 \
  --patience 3 \
  --add_nodes 5 \
  --test_percent 0.2  \
  --stride 1