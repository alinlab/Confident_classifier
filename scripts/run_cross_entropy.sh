# baseline 
export save=./results/cross_entropy/${RANDOM}/
mkdir -p $save
python ./src/run_cross_entropy.py --outf $save --dataroot ./data   2>&1 | tee  $save/log.txt
