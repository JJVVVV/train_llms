# wget 10.27.220.5:8025/baichuan_instruct.tar.gz
# tar -zxf baichuan_instruct.tar.gz

# wget 10.27.220.5:8201/sft_full_model.tar.gz
# tar -zxf sft_full_model.tar.gz

# wget 10.27.220.5:8201/baichuan_13b_sft_v2.tar
# tar xvf baichuan_13b_sft_v2.tar

# 安装依赖
pip config unset global.index-url
cd toolkit_pkg && pip install --editable . > ../training.log 2>&1 && cd -
pip install transformers_stream_generator
# pip install nltk
# pip install accelerate


# 定义快捷命令
echo "alias log='tail -f /root/paddlejob/workspace/env_run/training.log'" >> /root/.bashrc
echo "alias report='cat /root/paddlejob/workspace/env_run/outputs/report.log'" >> /root/.bashrc
echo "alias all_log='more /root/paddlejob/workspace/env_run/training.log'" >> /root/.bashrc
echo "alias ws='cd /root/paddlejob/workspace/env_run/'" >> /root/.bashrc
echo "alias killtrain='bash /root/paddlejob/workspace/env_run/bashScript/killtrain.sh'" >> /root/.bashrc
echo "alias update='# mv ~/codes/train_llms/train_llms.tar.gz ~/codes && cd ~/codes && find train_llms -maxdepth 1 -mindepth 1 ! -name runs ! -name outputs -exec rm -rf {} + && tar xzf train_llms.tar.gz && cd -'" >> /root/.bashrc

# 简短的bash提示符
echo "PS1=\"\[\e[36;1m\]\u\[\e[33;1m\]@\[\e[33;1m\]a100 \[\e[32;1m\]\W \[\e[31;1m\]$ \[\e[0m\]\"" >> /root/.bashrc

# 配置 ssh
sed -i '13i\Port 8000' /etc/ssh/sshd_config
sed -i '33i\PermitRootLogin yes' /etc/ssh/sshd_config
service ssh start
mkdir /root/.ssh && echo "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQCqTCJ0yfYWAq1i/CHtPCePRGvg0k/UA5ixnjMw1qkJMg3wngn+Bpaqptz1ACn6cX6TKFjF1439J/Nr4c0yvgwUis+3gZlmiz1LKseD4/C68wt63H/j6Je1sd5miiIAbPorJ9F9/wbn9NSP/I+e6GN0sV2UOYqsto2t9/+dbrrMKGBTU9NlwCqqjd8FYSC868PrVXTnKL2Ht/FQMbxTW2xK+2OKwV/8024M6Q9B79XhNEaffruj3n7gYqDeWlZYdcEwsICux8t8nMkmL1Pn+WRqZaqPvE+L6baTEW7a3n89llNPV/tbhMrtCTmbyFo+9hfwDlAyBrqvwn/nNQO5QcTp baidu" >> /root/.ssh/authorized_keys
# echo "1" | passwd --stdin  

# 禁止fire输出
sed -i '166,168 s/^/# /' /usr/local/python3.11.2/lib/python3.11/site-packages/fire/core.py

# 命名
mix_ratio=1
dataset_name="hot_finetune_data"
model_type="baichuan-13b-chat"
model_name="mix_general_data_ratio=$mix_ratio"

# 下载模型
if [ "$model_type" = "baichuan2-13b-chat" ]; then
    if [ -d "baichuan2-13b-chat" ]; then
        echo "Already downloaded."
    else
        echo "downloading..."
        wget 10.104.216.16:8202/baichuan2-13b-chat.tar > training.log 2>&1
        tar -xvf baichuan2-13b-chat.tar > training.log 2>&1
    fi
else
    if [ -d "baichuan-13b-chat" ]; then
        echo "Already downloaded."
    else
        echo "downloading..."
        wget 10.104.216.16:8201/baichuan-13b-chat.tar.gz > training.log 2>&1
        tar -zxvf baichuan-13b-chat.tar.gz > training.log 2>&1
    fi
fi

# 训练参数
pretrained_model_dir=./$model_type
# deepspeed_config_file=ds_zero2_no_offload.json
train_batch_size=64
infer_batch_size=64
gradient_accumulation_steps=4
# train_micro_batch_size_per_gpu=2
if [ "$model_type" = "baichuan2-13b-chat" ]; then
    fp16=False
    bf16=True
    torch_dtype=bfloat16
    deepspeed_config_file=./ds_zero3_offload2.hjson
    opt_lr="2e-4"
    sch_warmup_ratio_steps=0.2
else
    fp16=True
    bf16=False
    torch_dtype=float16
    deepspeed_config_file=./ds_zero3_offload.hjson
    opt_lr="1e-4"
    sch_warmup_ratio_steps=0.03
fi
epochs=10
opt_weight_decay=0.01
train_file_path=./data/$dataset_name/train/mixed_ratio=$mix_ratio.jsonl
val_file_path=./data/$dataset_name/val/all.json


# 推理参数
max_new_tokens=1024
temperature=0.01
top_k=5
top_p=0.85
do_sample=True
num_beams=1
repetition_penalty=1.1

torchrun --nnodes 1 --nproc_per_node 8 train.py \
    --model_name $model_name \
    --model_type $model_type \
    --dataset_name $dataset_name \
    --cache_dir "None" \
    --model_revision "main" \
    --use_fast_tokenizer "True" \
    --use_auth_token "False" \
    --torch_dtype $torch_dtype \
    --preprocessing_num_workers 8 \
    --max_seq_length 2048 \
    --parallel_mode "deepspeed" \
    --dashboard "tensorboard" \
    --deepspeed_config ${deepspeed_config_file} \
    --metric "rougeL" \
    --model_dir ${pretrained_model_dir} \
    --train_file_path ${train_file_path} \
    --val_file_path ${val_file_path} \
    --train_batch_size ${train_batch_size} \
    --infer_batch_size ${infer_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --seed 0 \
    --fp16 $fp16 \
    --bf16 $bf16 \
    --epochs $epochs \
    --opt_lr ${opt_lr} \
    --sch_warmup_ratio_steps $sch_warmup_ratio_steps \
    --opt_weight_decay $opt_weight_decay \
    --ddp_timeout 30000 \
    --logging_steps 5 \
    --padding_side "left" \
    --max_new_tokens $max_new_tokens \
    --temperature $temperature \
    --top_k $top_k \
    --top_p $top_p \
    --do_sample $do_sample \
    --num_beams $num_beams \
    --repetition_penalty $repetition_penalty \
    --save_dir None \
    --cut_input_from_output True \
    --generate_config_file "generate_config.json" \
    --re_gen_num 2 \
    --use_deepspeed_ckpt False \
    --save_all_ckpts True \
    > training.log 2>&1

    # --seed $RANDOM \

    # --train_micro_batch_size_per_gpu ${train_micro_batch_size_per_gpu} \
    # --gradient_accumulation_steps ${gradient_accumulation_steps} \
    # --lr_scheduler_type cosine \
    # --gradient_checkpointing \
