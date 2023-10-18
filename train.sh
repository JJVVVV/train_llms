# wget 10.27.220.5:8025/baichuan_instruct.tar.gz
# tar -zxf baichuan_instruct.tar.gz

# wget 10.27.220.5:8201/sft_full_model.tar.gz
# tar -zxf sft_full_model.tar.gz

# wget 10.27.220.5:8201/baichuan_13b_sft_v2.tar
# tar xvf baichuan_13b_sft_v2.tar

# 安装依赖
pip config unset global.index-url
cd toolkit_pkg && pip install --editable . > ../pip.log 2>&1 && cd -
pip install transformers_stream_generator


# 下载模型
wget 10.104.216.16:8201/baichuan-13b-chat.tar.gz
tar -zxf baichuan-13b-chat.tar.gz

# 定义快捷命令
echo "alias log='tail -f /root/paddlejob/workspace/env_run/training.log'" >> /root/.bashrc
echo "alias report='cat /root/paddlejob/workspace/env_run/outputs/report.log'" >> /root/.bashrc
echo "alias all_log='more /root/paddlejob/workspace/env_run/training.log'" >> /root/.bashrc
echo "alias ws='cd /root/paddlejob/workspace/env_run/'" >> /root/.bashrc
echo "alias killtrain='bash /root/paddlejob/workspace/env_run/bashScript/killtrain.sh'" >> /root/.bashrc
echo "alias update='cd ~/codes && find train_llms -maxdepth 1 -mindepth 1 ! -name runs ! -name outputs -exec rm -rf {} + && tar xzf train_llms.tar.gz && cd -'" >> /root/.bashrc

# 简短的bash提示符
echo "PS1=\"\[\e[36;1m\]\u\[\e[33;1m\]@\[\e[33;1m\]a100 \[\e[32;1m\]\W \[\e[31;1m\]$ \[\e[0m\]\"" >> /root/.bashrc

# 配置 ssh
service ssh start
# echo "1" | passwd --stdin  

# 禁止fire输出
sed -i '166,168 s/^/# /' /usr/local/python3.11.2/lib/python3.11/site-packages/fire/core.py

# 创建运行时输出目录
mkdir ~/running

# 训练参数
pretrained_model_dir=./baichuan-13b-chat
# deepspeed_config_file=ds_zero2_no_offload.json
deepspeed_config_file=./ds_zero3_offload.hjson
train_batch_size=64
infer_batch_size=64
gradient_accumulation_steps=4
# train_micro_batch_size_per_gpu=2
fp16=True
bf16=False
epochs=10
opt_lr=1e-4
opt_weight_decay=0
sch_warmup_ratio_steps=0.03
train_file_path=./data/hot_finetune_data/train
val_file_path=./data/hot_finetune_data/val/all.json

# 推理参数
max_new_tokens=1024
temperature=0.01
top_k=5
top_p=0.85
do_sample=True
num_beams=1
repetition_penalty=1.1

torchrun --nnodes 1 --nproc_per_node 8 train.py \
    --cache_dir "None" \
    --model_revision "main" \
    --use_fast_tokenizer "True" \
    --use_auth_token "False" \
    --torch_dtype "float16" \
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
    --seed $RANDOM \
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
    --save_dir "outputs" \
    --save_last_ckpt False \
    --generate_config_file "generate_config.json" \
    --re_gen_num 2 \
    --use_deepspeed_ckpt False \
    --save_all_ckpts True \
    > training.log 2>&1

    # --train_micro_batch_size_per_gpu ${train_micro_batch_size_per_gpu} \
    # --gradient_accumulation_steps ${gradient_accumulation_steps} \
    # --save_strategy steps \
    # --save_steps 5 \
    # --save_total_limit 5 \
    # --evaluation_strategy steps \
    # --eval_steps 1000000 \
    # --lr_scheduler_type cosine \
    # --gradient_checkpointing \
    # --ddp_find_unused_parameters False \
    # --max_seq_length 2048 \
