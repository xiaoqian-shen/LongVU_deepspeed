
PREV_STAGE_CHECKPOINT="./checkpoints/cambrian_qwen_7b"
PATH_TO_JSON="/home/shenx/videos/LongVU/data.json"
PATH_TO_FOLDER="./mlvu"
VERSION="qwen"

CUDA_VISIBLE_DEVICE=3 CUDA_LAUNCH_BLOCKING=4 TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node=1 --nnodes=1 \
train.py \
--deepspeed zero3.json \
--output_dir "/tmp/longvu/" \
--input_model_filename $PREV_STAGE_CHECKPOINT \
--output_model_filename "./checkpoints/cambrian_qwen/" \
--data_path $PATH_TO_JSON \
--image_folder $PATH_TO_FOLDER \
--model_max_length 8192 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--logging_dir /tmp/llava/test/ \
--num_train_epochs 1 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--save_steps 500 \
--eval_steps 500 \
--logging_steps 10 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--report_to "tensorboard" \
--save_total_limit 1 \
--learning_rate 5e-6 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--tf32 False \
--version $VERSION \
--mm_vision_select_layer "-2" \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--image_aspect_ratio pad \
--group_by_modality_length True \
--dataloader_num_workers 0 \
--lazy_preprocess True \
--tune_mm_mlp_adapter False \
--freeze_mm_mlp_adapter False \
--freeze_backbone False \
--gradient_checkpointing True \
--mm_projector_type sva \
--image_token_len 144 \
--query_num_list "[144]" \
--resume True \
--lowres_token 8 \
--video_fps 1 \
--highres True \
--drop_threshold 0.8 \
# --fsdp "full_shard auto_wrap" \
# --fsdp_transformer_layer_cls_to_wrap 'Qwen2DecoderLayer' \
