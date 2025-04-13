import torch
import torch.distributed as dist
import argparse
import multiprocessing as mp
import socket
import os
import fairscale.nn.model_parallel.initialize as fs_init
import json
import sys
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import functools
import numpy as np
import torchvision
from torch import nn
sys.path.append("/data3/xy/LlamaGen-origin/tokenizer")
from tokenizer_image.vq_model import VQ_models
import time
import math
from tqdm import tqdm
from Network.gpt import GPT_models
import os
os.environ["CUDA_VISIBLE_DEVICES"] ='7'

def find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

def adap_sche(step, mode="arccos", leave=False):
    """ Create a sampling scheduler
        :param
        step  -> int:  number of prediction during inference
        mode  -> str:  the rate of value to unmask
        leave -> bool: tqdm arg on either to keep the bar or not
        :return
        scheduler -> torch.LongTensor(): the list of token to predict at each step
    """
    r = torch.linspace(1, 0, step)
    if mode == "root":              # root scheduler
        val_to_mask = 1 - (r ** .5)
    elif mode == "linear":          # linear scheduler
        val_to_mask = 1 - r
    elif mode == "square":          # square scheduler
        val_to_mask = 1 - (r ** 2)
    elif mode == "cosine":          # cosine scheduler
        val_to_mask = torch.cos(r * math.pi * 0.5)
    elif mode == "arccos":          # arc cosine scheduler
        val_to_mask = torch.arccos(r) / (math.pi * 0.5)
    else:
        return

    # fill the scheduler by the ratio of tokens to predict at each step
    sche = (val_to_mask / val_to_mask.sum()) * (256)
    sche = sche.round()
    sche[sche == 0] = 1                                                  # add 1 to predict a least 1 token / step
    sche[-1] += (256) - sche.sum()         # need to sum up nb of code
    return tqdm(sche.int(), leave=leave)

def adap_sche(step, mode="arccos", leave=False):
    """ Create a sampling scheduler
        :param
        step  -> int:  number of prediction during inference
        mode  -> str:  the rate of value to unmask
        leave -> bool: tqdm arg on either to keep the bar or not
        :return
        scheduler -> torch.LongTensor(): the list of token to predict at each step
    """
    r = torch.linspace(1, 0, step)
    if mode == "root":              # root scheduler
        val_to_mask = 1 - (r ** .5)
    elif mode == "linear":          # linear scheduler
        val_to_mask = 1 - r
    elif mode == "square":          # square scheduler
        val_to_mask = 1 - (r ** 2)
    elif mode == "cosine":          # cosine scheduler
        val_to_mask = torch.cos(r * math.pi * 0.5)
    elif mode == "arccos":          # arc cosine scheduler
        val_to_mask = torch.arccos(r) / (math.pi * 0.5)
    else:
        return

    # fill the scheduler by the ratio of tokens to predict at each step
    sche = (val_to_mask / val_to_mask.sum()) * (256)
    sche = sche.round()
    sche[sche == 0] = 1                                                  # add 1 to predict a least 1 token / step
    sche[-1] += (256) - sche.sum()         # need to sum up nb of code
    return tqdm(sche.int(), leave=leave)

def main(args, rank, master_port):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(args.num_gpus)
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["MASTER_ADDR"] = "127.0.0.1"

    dist.init_process_group("nccl")
    fs_init.initialize_model_parallel(1)
    device = "cuda"
    # seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(args.global_seed)
    torch.cuda.set_device(rank)

    # create and load tokenizer
    vq_model = VQ_models['VQ-16'](codebook_size=16384, codebook_embed_dim=8)
    vq_model.eval()
    checkpoint = torch.load('/data3/xy/vq_ds16_c2i.pt', map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"], strict=True)
    vq_model = vq_model.to(device)
    del checkpoint
    print(f"Image tokenizer is loaded")

    # create and load DiT model
    torch_dtype = {"fp32": torch.float, "tf32": torch.float, "bf16": torch.bfloat16, "fp16": torch.float16,}[args.precision]
    latent_size = args.image_size // 16
    """
        load llamagen
    """
    model = GPT_models["GPT-B"](
            vocab_size=args.codebook_size,
            block_size=256,
            num_classes=1000,
            cls_token_num=1,
            model_type='c2i'
        )

    ckpt_path = '/data3/xy/llamagen-llamagen/pretrained_maskgit/MaskGITepoch_050.pth'
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model = model.to(device)
    model.eval()
    del checkpoint
    print(f"Model is loaded")


    # save_folder = "evaluate"
    batch_size = 100
    num_steps = args.num_images // (batch_size * dist.get_world_size()) + 1
    class_label_gen_world = np.arange(0, args.num_classes).repeat(args.num_images // args.num_classes)
    # print(class_label_gen_world)
    # class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])
    print(len(class_label_gen_world))
    # world_size = dist.get_world_size()
    # local_rank = dist.get_rank()
    # used_time = 0
    gen_img_cnt = 0
    init_code = None
    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))
        l_codes = []  # Save the intermediate codes predicted
        l_mask = []   # Save the intermediate masks
        with torch.no_grad():
            # with torch.cuda.amp.autocast():
            begin = batch_size * i
            end = batch_size * (i+1)
            labels_gen = class_label_gen_world[begin:end]
            labels_gen = torch.Tensor(labels_gen).long().cuda()
            drop = torch.ones(len(labels_gen), dtype=torch.bool).to(device)
            if init_code is not None:
                code = init_code
                mask = (init_code == args.codebook_size).float().view(len(labels_gen), 256)
            else:  # Initialize a code 
                code = torch.full((len(labels_gen), 16, 16), args.codebook_size).to(device)
                mask = torch.ones((len(labels_gen), 256)).to(device)
            scheduler = adap_sche(step=args.inference_iters, mode='arccos')
            for indice, t in enumerate(scheduler):
                # print(t)
                if mask.sum() < t:  # Cannot predict more token than 16*16 or 32*32
                    t = int(mask.sum().item())

                if mask.sum() == 0:  # Break if code is fully predicted
                    break

                # with torch.cuda.amp.autocast():  # half precision
                if args.cfg_scale != 0:
                    # Model Prediction
                    logit = model(torch.cat([code.clone(), code.clone()], dim=0),torch.cat([labels_gen, labels_gen], dim=0), torch.cat([~drop, drop], dim=0))[:, :, :-1]
                    print(logit.shape)
                    logit_c, logit_u = torch.chunk(logit, 2, dim=0)
                    _w = args.cfg_scale * (indice / (len(scheduler)-1))
                    # Classifier Free Guidance
                    logit = (1 + _w) * logit_c - _w * logit_u
                else:
                    logit = model(code.clone(), labels_gen, drop_label=~drop)  
                    
                prob = torch.softmax(logit * args.temperature, -1)
                # Sample the code from the softmax prediction
                distri = torch.distributions.Categorical(probs=prob)
                pred_code = distri.sample()

                conf = torch.gather(prob, 2, pred_code.view(len(labels_gen), 256, 1))

                if args.randomize == "linear":  # add gumbel noise decreasing over the sampling process
                    ratio = (indice / (len(scheduler)-1))
                    rand = args.r_temp * np.random.gumbel(size=(len(labels_gen), 256)) * (1 - ratio)
                    conf = torch.log(conf.squeeze()) + torch.from_numpy(rand).to(device)
                elif args.randomize == "warm_up":  # chose random sample for the 2 first steps
                    conf = torch.rand_like(conf) if indice < 2 else conf
                elif args.randomize == "random":   # chose random prediction at each step
                    conf = torch.rand_like(conf)

                # do not predict on already predicted tokens
                conf[~mask.bool()] = -math.inf

                # chose the predicted token with the highest confidence
                tresh_conf, indice_mask = torch.topk(conf.view(len(labels_gen), -1), k=t, dim=-1)
                tresh_conf = tresh_conf[:, -1]

                # replace the chosen tokens
                conf = (conf >= tresh_conf.unsqueeze(-1)).view(len(labels_gen), 16, 16)
                f_mask = (mask.view(len(labels_gen), 16, 16).float() * conf.view(len(labels_gen), 16, 16).float()).bool()
                code[f_mask] = pred_code.view(len(labels_gen), 16, 16)[f_mask]
                # update the mask
                for i_mask, ind_mask in enumerate(indice_mask):
                    mask[i_mask, ind_mask] = 0
                l_codes.append(pred_code.view(len(labels_gen), 16, 16).clone())
                l_mask.append(mask.view(len(labels_gen), 16, 16).clone())
            print(code.max())
            _code = torch.clamp(code, 0,  args.codebook_size-1)
            qzshape = [_code.shape[0], args.codebook_embed_dim, 16, 16]
            samples = vq_model.decode_code(code, qzshape)
            for i in range(0, len(labels_gen)):
                # 获取当前图片数据
                img_data = samples[i]

                # 归一化到 [0, 1]
                img_normalized = (img_data - img_data.min()) / (img_data.max() - img_data.min())

                # 转换为 [0, 255] 的像素值
                img_array = (img_normalized * 255).detach().cpu().to(torch.uint8).numpy() 

                # 将tensor转换为PIL图片，注意颜色通道需要转到最后一个维度
                img = Image.fromarray(img_array.transpose(1, 2, 0))
                # 可以保存图片
                img.save(f'evaluate/image_{gen_img_cnt}.png')
                gen_img_cnt = gen_img_cnt + 1                       
        # break

    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument("--model", type=str, default="DiT_Llama_600M_patch2")
    parser.add_argument("--model", type=str, choices=list(GPT_models.keys()), default="GPT-L")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional") 
    parser.add_argument("--model-ckpt", type=str, default="results/DiT_Llama_600M_patch2_bs512_lr1e-4_bf16_qknorm_lognorm/checkpoints/0110000")
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default="LlamaGen/vq_ds16_c2i.pt", help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--qk-norm", action="store_true",)
    parser.add_argument("--num_images", type=int, default=50000)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", type=str, choices=["fp32", "tf32", "fp16", "bf16"],default="bf16",)
    parser.add_argument("--image_size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--inference_iters", type=int, default=32)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    
    parser.add_argument("--randomize", type=str, default='linear')
    # parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    # parser.add_argument("--from-fsdp", action='store_true')
    # parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    # parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=384)
    # parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    # parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--r_temp", type=float, default=4.5)
    # parser.add_argument("--top-k", type=int, default=2000,help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    # parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    args = parser.parse_args()

    master_port = find_free_port()
    mp.set_start_method("spawn")
    procs = []
    for i in range(args.num_gpus):
        p = mp.Process(target=main, args=(args, i, master_port))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()