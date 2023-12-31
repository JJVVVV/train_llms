# nohup ./train_pure.sh > /dev/null 2>&1 &
# tmux new-session -d -s train_task 'bash ./train_pure.sh > /dev/null 2>&1'

# 命名
# mix_ratio=0.5
dataset_name="hot_finetune_data"
model_type="baichuan-13b-chat"
# model_name="mix_general_data_ratio=$mix_ratio"
model_name="sample_context_with_aug"

# 训练参数
pretrained_model_dir='/root/paddlejob/workspace/env_run/baichuan-13b-chat'
# deepspeed_config_file=ds_zero2_no_offload.json
train_batch_size=64
infer_batch_size=16
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
# train_file_path=./data/$dataset_name/train/mixed_ratio=$mix_ratio.jsonl
train_file_path=./data/hot_finetune_data/train/with_aug.jsonl
val_file_path=./data/$dataset_name/val/all.json


# 推理参数
max_new_tokens=1024
temperature=0.01
top_k=5
top_p=0.85
do_sample=True
num_beams=1
repetition_penalty=1.1

# --master_port=25678
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
    --logging_steps 1 \
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