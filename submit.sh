#!/usr/bin bash
set -ex

cur_time=`date  +"%Y%m%d%H%M"`
job_name=train_nlp_text-generation_baichuan_sft_${cur_time}

## k8s cluster
group_name="kg-40g-1-yq01-k8s-gpu-a100-8-0"

job_version="paddle-fluid-custom"

# image_addr="iregistry.baidu-int.com/huzhe01/huzhe01-transformers-pdc:deepspeed"
image_addr="iregistry.baidu-int.com/jjw/dev:2"

start_cmd="sh train.sh"
# start_cmd="sh run_rm.sh"
k8s_trainers=1
k8s_gpu_cards=8
wall_time="1000:00:00"

k8s_priority="normal"
file_dir="."

job_tags="cg-13b-instruct-finetune"
job_remark=$job_tags

ak=aa91e5f9a53e595d97ed073e17e13e90
sk=046bab824c3b52bc8fcdfc7526d9e794

# if [ $2 -ne *"history"* ]; then
# cp $2 ./history/$1.$2.$3
# fi
        # --use-native-api 0 --dfs-agent-port 21270 \
~/paddlecloud-cli/paddlecloud job --ak ${ak} --sk ${sk} \
        train \
        --job-name ${job_name} \
        --job-conf config.ini \
        --group-name ${group_name} \
        --start-cmd "${start_cmd}" \
        --file-dir ${file_dir} \
        --job-version ${job_version}  \
        --job-tags ${job_tags} \
        --image-addr ${image_addr} \
        --job-remark ${job_remark} \
        --k8s-trainers ${k8s_trainers} \
        --k8s-gpu-cards ${k8s_gpu_cards} \
        --k8s-priority ${k8s_priority} \
        --wall-time ${wall_time} \
        --is-standalone 1 \
        --algo-id algo-476cfcbeec5a4739
