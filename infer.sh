# CUDA_VISIBLE_DEVICES=3 python infer.py epoch2
# CUDA_VISIBLE_DEVICES=3 python infer.py tmp
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python infer.py design-13w-ldm-sr 2022-12-12T13-46-40 epoch=000079.ckpt compare
for a in 0 1 2 3 4 5 6 7
do 
    CUDA_VISIBLE_DEVICES=$a python infer.py --model_log_dir design-13w-ldm-sr \
        --config_name 2022-12-12T13-46-40 \
        --ckpt_name epoch=000079.ckpt \
        --dirs compare$a &
done
# python -m torch.distributed.launch --nproc_per_node=8 \
#     infer.py --model_log_dir design-13w-ldm-sr \
#     --config_name 2022-12-12T13-46-40 \
#     --ckpt_name epoch=000079.ckpt \
#     --dirs compare
# 