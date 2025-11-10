# -*- coding: utf-8 -*-   
# Author       : Chongyang Li
# Email        : lichongyang2016@163.com
# Date         : 2025-11-10 17:25:06
# LastEditors  : Chongyang Li
# LastEditTime : 2025-11-10 20:22:41
# FilePath     : /BPG_LDPC/utils.py
import matplotlib.pyplot as plt
import torch
import numpy as np
import subprocess
import tensorflow as tf
from glob import glob
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os

plt.rcParams["savefig.bbox"] = 'tight'


def save_image(raw_img, de_img, name):
        raw_img = raw_img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        de_img = de_img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        raw_img = raw_img / 255. if raw_img.max() > 1.1 else raw_img
        de_img = de_img / 255. if de_img.max() > 1.1 else de_img

        plt.figure(figsize=(7,15))

        plt.subplot(2,1,1)
        plt.imshow(raw_img)
        plt.axis('off')

        plt.subplot(2,1,2)
        plt.imshow(de_img)
        plt.axis('off')

        plt.savefig('%s.png' % name)
        plt.close()

def dec2bin(x, bits):
    # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float().flatten()


def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)

def snrdb2no(snr_db, num_bits_per_symbol, coderate):
    ebno_db = snr_db - 10 * np.log10(num_bits_per_symbol) - 10 * np.log10(coderate)
    return ebno_db

def bpg_compress(input_file, output_file, quality=30, bit_depth=8, color_space='ycbcr', extra_options=None):
    """
    调用 bpgenc，把 input_file 压缩成 BPG 存到 output_file。
    :param input_file: 输入图像路径，如 'input.png'
    :param output_file: 输出 BPG 文件路径，如 'output.bpg'
    :param quality: bpgenc 的 -q 参数，值越大质量越差，文件也越小
    :param bit_depth: bpgenc 的 -b 参数，默认 8 位，最大可到 14
    :param color_space: bpgenc 的 -c 参数，可选 'ycbcr', 'rgb', 'ycgco' 等
    :param extra_options: 列表，包含要额外传给 bpgenc 的其他选项
    """
    if extra_options is None:
        extra_options = []
    
    # bpgenc 常见选项:
    #   -o <outfile> : 指定输出文件
    #   -q <quality> : 质量 [0..51]
    #   -b <bit_depth>: 指定位深度 [8..14]
    #   -c <color_space> : 指定色彩空间，ycbcr(默认)/rgb/ycgco
    #   -f <format> : 指定像素格式(4:4:4,4:2:2,4:2:0等)
    #   -m <level>  : HEVC 编码复杂度 [0..9], 数值越大压缩速度越慢但效果更好
    #   -lossless   : 无损压缩(此时 -q,-b 可能失效，需同时指定rgb)
    #   其他更多选项见 bpgenc -h

    cmd = [
        'bpgenc',
        '-o', output_file,
        '-q', str(quality),
        '-b', str(bit_depth),
        '-c', color_space
    ] + extra_options + [input_file]
    
    # print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    # print("BPG compression done. Output =>", output_file)

def bpg_decompress(input_bpg, output_file, extra_options=None):
    """
    调用 bpgdec，把 BPG 文件解压为普通图像，如 PNG/PPM/PGM 等。
    :param input_bpg: BPG 格式文件路径
    :param output_file: 解压后的图像文件路径
    :param extra_options: 其他选项
    """
    if extra_options is None:
        extra_options = []
    
    # bpgdec 常见选项:
    #   -o <outfile>: 指定输出文件
    #   -b <bit_depth>: 输出时使用多少位
    #   -f <format> : 输出图像格式： 'png'/'ppm'/'pgm'
    #   -alpha : 单独输出 alpha 通道
    #   -csky : (很少用)
    #   其他更多选项见 bpgdec -h

    cmd = [
        'bpgdec',
        '-o', output_file
    ] + extra_options + [input_bpg]

    # print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    # print("BPG decompression done. Output =>", output_file)

def split_into_blocks(bits_tensor, max_k):
    """将比特流分割成最大为max_k的小块"""
    batch_size = tf.shape(bits_tensor)[0]
    num_bits = bits_tensor.shape[1]
    
    num_blocks = (num_bits + max_k - 1) // max_k
    total_bits = num_blocks * max_k
    
    # padding 到完整的 block 数
    padded = tf.pad(bits_tensor, [[0,0],[0, total_bits - num_bits]])
    # reshape 成 [batch*num_blocks, max_k]
    blocks = tf.reshape(padded, [batch_size*num_blocks, max_k])
    return blocks, num_bits, num_blocks


class Datasets(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.imgs = []
        for dir in self.data_dir:
            self.imgs += glob(os.path.join(dir, '*.jpg'))
            self.imgs += glob(os.path.join(dir, '*.png'))
        self.imgs.sort()


    def __getitem__(self, item):
        image_ori = self.imgs[item]
        image = Image.open(image_ori).convert('RGB')
        self.im_height, self.im_width = image.size
        if self.im_height % 128 != 0 or self.im_width % 128 != 0:
            self.im_height = self.im_height - self.im_height % 128
            self.im_width = self.im_width - self.im_width % 128
        self.transform = transforms.Compose([
            transforms.CenterCrop((self.im_width, self.im_height)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 255).byte())])
        transformed = self.transform(image)
        # transforms.ToPILImage()(transformed).save('test.png')
        return transformed
    def __len__(self):
        return len(self.imgs)

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    def clear(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0