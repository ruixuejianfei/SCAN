now=$(date +"%Y%m%d_%H%M%S")

export PATH=/mnt/lustre/lijingyu/Data_t1/anaconda2/envs/py27pt02/bin:$PATH
export TORCH_MODEL_ZOO=/mnt/lustre/DATAshare2/sunhongbin/pytorch_pretrained_models

split=0
jobname=ilid-$split-scan-128

num_gpus=4
log_dir=logs/ilid-split${split}-scan-256

if [ ! -d $log_dir ]; then
  echo create log $log_dir
  mkdir -p $log_dir
fi

srun -p TITANXP --job-name=$jobname --gres=gpu:$num_gpus  \
python  -u train_val.py \
        -d ilidsvidsequence \
        -b 32 \
        --seq_len 10 \
        --seq_srd 5 \
        --split $split \
        --epoch 30 \
        --features 128 \
        --a1 resnet50 \
        --lr1 1e-3 \
        --lr2 1e-3 \
        --lr3 1 \
        --train_mode cnn_rnn \
        --lr1step 20 \
        --lr2step 10 \
        --lr3step 30 \
        --logs-dir $log_dir \
        2>&1 | tee ${log_dir}/record-train-${now}.txt &\
