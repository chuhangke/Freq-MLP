nohup /home/kch/科研/LargeST-main/experiments/fits/main.py --device cuda:0 --dataset SD --years 2019_48 --model_name fits_sd --seed 2024 --bs 64 --max_epoch 100 --input_dim 1> fits_sd.log 2>&1
nohup /home/kch/科研/LargeST-main/experiments/fits/main.py --device cuda:0 --dataset GBA --years 2019_48 --model_name fits_gba --seed 2024 --bs 64 --max_epoch 100 --input_dim 1> fits_gba.log 2>&1
nohup python /home/kch/科研/LargeST-main/experiments/fits/main.py --device cuda:0 --dataset GLA --years 2019_48 --model_name fits_gla --seed 2024 --bs 64 --max_epoch 100 --input_dim 1> fits_gla.log 2>&1
nohup python /home/kch/科研/LargeST-main/experiments/fits/main.py --device cuda:0 --dataset CA --years 2019_48 --model_name fits_ca --seed 2024 --bs 64 --max_epoch 100 --input_dim 1> fits.log 2>&1