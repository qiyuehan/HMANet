

for pred_len in 96 192 336 720
do
  python -u LSTF_Forecasting.py \
  --data ETTh1.csv \
  --model_name ETTh1 \
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


  python -u LSTF_Forecasting.py \
  --data ETTh2.csv \
  --model_name ETTh2 \
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

  python -u LSTF_Forecasting.py \
  --data ETTm1.csv \
  --model_name ETTm1 \
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

  python -u LSTF_Forecasting.py \
  --data ETTm2.csv \
  --model_name ETTm2 \
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

done



