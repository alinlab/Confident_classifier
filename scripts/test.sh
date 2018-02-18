# baseline 
export save=./test/${RANDOM}/
mkdir -p $save
python ./src/measure_confidence.py --outf $save --dataset $1 --nt_dataset $2 --pre_trained_net $3  --temper $4 --noi $5 --dataroot ./data   2>&1 | tee  $save/log.txt
