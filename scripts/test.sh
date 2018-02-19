# baseline 
export save=./test/${RANDOM}/
mkdir -p $save
python ./src/test_detection.py --outf $save --dataset $1 --out_dataset $2 --pre_trained_net $3  --dataroot ./data   2>&1 | tee  $save/log.txt
