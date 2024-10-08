data_folder="/data0/data/imagenet"  # imagenet data path
vit_folder="./pretrained_maskgit/MaskGIT" # no change
vqgan_folder="./pretrained_maskgit/VQGAN/" # no change
writer_log="./logs/" # no change
num_worker=16
gpt_model='GPT-B' # scaling--> GPT-L (300M), GPT-XL (700M), GPT-XXL (1.4B)
bsize=2  # change with GPU nums.   192 // GPUs

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --standalone --nnodes=1 --nproc_per_node=6 main.py  \
        --bsize ${bsize} --data-folder "${data_folder}" --vit-folder "${vit_folder}" \
        --vqgan-folder "${vqgan_folder}" --writer-log "${writer_log}" \
        --num_workers ${num_worker} --img-size 256 --epoch 301 \
        --gpt_model "${gpt_model}" 

