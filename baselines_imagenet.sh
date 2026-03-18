CUDA_VISIBLE_DEVICES=3 python src/main.py --name FedMMDP-avg --FL_algorithm FedAvg --lr 1e-5 --local_epochs 1 --comm_rounds 20  --batch_size 256 --model clip >outputs/output_avg_clip.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python src/main.py --name FedMMDP-prox --FL_algorithm FedProx --lr 1e-5 --local_epochs 1 --comm_rounds 20 --model clip --batch_size 256 >outputs/output_prox.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python src/main.py --name FedMMDP-md --FL_algorithm FedMD --lr 1e-5 --local_epochs 1 --comm_rounds 20 --model clip --batch_size 256 --pub_data_num 5000 >outputs/output_md.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python src/main.py --name FedMMDP-df --FL_algorithm FedDF --lr 1e-5 --local_epochs 1 --comm_rounds 20 --model clip --batch_size 256 --pub_data_num 5000 >outputs/output_df.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python src/main.py --name FedMMDP-Cream --FL_algorithm Cream --lr 1e-5 --local_epochs 1 --comm_rounds 20 --model clip --batch_size 256 --pub_data_num 5000 >outputs/output_cream.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python src/main.py --name FedMMDP-Harmony --FL_algorithm Harmony --lr 1e-5 --local_epochs 1 --comm_rounds 20 --model clip --batch_size 256 >outputs/output_Harmony.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python src/main.py --name FedMMDP-MASA --FL_algorithm MASA --lr 1e-5 --local_epochs 1 --comm_rounds 20 --model clip --batch_size 256 >outputs/output_MASA.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python src/main.py --name FedMMDP-MEMA --FL_algorithm FedMEMA --lr 1e-5 --local_epochs 1 --comm_rounds 20 --model clip --batch_size 256 >outputs/output_FedMEMA.log 2>&1 &