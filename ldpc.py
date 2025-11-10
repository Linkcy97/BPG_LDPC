# -*- coding: utf-8 -*-   
# Author       : Chongyang Li
# Email        : lichongyang2016@163.com
# Date         : 2024-10-10 08:35:53
# LastEditors  : Chongyang Li
# LastEditTime : 2025-11-10 20:22:18
# FilePath     : /BPG_LDPC/ldpc.py

from PIL import Image
import torch
from tqdm import tqdm
import numpy as np
from utils import *
import os, sys
import logging
os.chdir(sys.path[0])
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sionna
from sionna.phy.mimo import lmmse_equalizer
from sionna.phy.mapping import Demapper, Mapper, Constellation, BinarySource
from sionna.phy.utils import ebnodb2no
from sionna.phy.channel import FlatFadingChannel, AWGN
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder
from sionna.phy.utils.metrics import compute_ber

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from torchvision import io, datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from distortion import *

torch.manual_seed(0)
seed = np.random.RandomState(42)

CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3)

    
# 将 PIL 图像转换为 Tensor
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为 [0, 1] 范围的浮点数
    transforms.Lambda(lambda x: (x * 255).byte())  # 转换为 [0, 255] 范围的 uint8
])


test_dataset = datasets.CIFAR10(root="/home/linkcy97/Datasets/CIFAR10/",
                                train=False,
                                transform=transform,
                                download=False)
# indices = np.random.choice(len(test_dataset), 100, replace=False)
indices = list(range(9000, 9050))
subset_dataset = Subset(test_dataset, indices)

kodak_dataset = Datasets(["/home/linkcy97/Datasets/Kodak/"])

test_loader = DataLoader(dataset=subset_dataset,
                            batch_size=1,
                            shuffle=False)
test_loader_all = DataLoader(dataset=test_dataset,
                            batch_size=1,
                            shuffle=False)
kodak_loader = DataLoader(dataset=kodak_dataset,
                            batch_size=1,
                            shuffle=False)


# Communication System Parameters
demapping_method = "app" # try "max-log"
ldpc_cn_type = "boxplus" # try also "minsum"
nbits = 2                # number of bits mapped to one symbol (cf. QAM)
code_rate = 1/4          # LDPC code rate  
max_k = 3840             # maximum number of information bits per LDPC codeword 1/4: 3840, 1/2: 8448
constellation = Constellation("qam", nbits)
mapper = Mapper(constellation=constellation)
multiple_snr = [7]             # set of SNR points to simulate
channel_type = 'rayleigh'  # 'awgn' or 'rayleigh'
if channel_type == 'awgn':
    channel = AWGN()
else:
    channel = FlatFadingChannel(1, 1, add_awgn=True, return_channel=True)
demapper = Demapper(demapping_method, constellation=constellation)

def ldpc_encode_blocks(blocks, ldpc_rate):
    """
    对 [batch*num_blocks, k] 的输入并行编码
    """
    k = blocks.shape[1]
    n = int(k / ldpc_rate)

    encoder = LDPC5GEncoder(k, n)
    encoded = encoder(blocks)  # shape: [batch*num_blocks, n]

    return encoded, k, n

def ldpc_decode_blocks(received_blocks, k, n, num_bits, num_blocks):
    """
    并行解码，再拼接成原始比特流
    received_blocks: [batch*num_blocks, n]
    """
    encoder = LDPC5GEncoder(k, n)
    decoder = LDPC5GDecoder(encoder, hard_out=True)

    decoded = decoder(received_blocks)  # [batch*num_blocks, k]

    # reshape 回原始 [batch, num_blocks*k]
    batch_size = tf.shape(received_blocks)[0] // num_blocks
    decoded = tf.reshape(decoded, [batch_size, num_blocks*k])

    # 截断 padding 部分，恢复到原始 num_bits
    decoded = decoded[:, :num_bits]

    return decoded

def test_bpg_ldpc_channel_single_image(
    raw_img_tensor,
    bpg_quality=30,
    ldpc_rate=0.2,
    max_k=3840,
    ebno_db=6.0,
    channel_type=channel_type):
    """
    对单张图执行:
      1) 保存PNG -> 调用bpgenc压缩 -> 读入BPG字节流
      2) BPG比特流 -> LDPC + 信道 -> BPG比特流
      3) 写回BPG文件 -> bpgdec 解码 -> 计算PSNR
    :param raw_img_tensor: 形状(C,H,W)，像素范围[0,255]的uint8 Tensor
    :param bpg_quality: bpgenc的 -q 参数
    :param ldpc_rate: LDPC 编码码率
    :param ebno_db: 信道Eb/No (dB)
    :param channel_type: 'awgn' 或 'rayleigh'
    :return: psnr, cbr
    """
    temp_input_png = "temp_input.png"
    temp_bpg = "temp_output.bpg"
    temp_bpg_received = "temp_received.bpg"
    temp_recon_png = "temp_recon.png"

    # 保存图像为PNG文件
    pil_img = transforms.ToPILImage()(torch.squeeze(raw_img_tensor, 0))
    pil_img.save(temp_input_png)
    # 1) 调用 bpgenc 压缩成 BPG 文件
    bpg_compress(
        input_file=temp_input_png,
        output_file=temp_bpg,
        quality=bpg_quality,
        bit_depth=8,
        color_space='ycbcr',
        extra_options=['-f','444']  # 比如强制4:4:4
    )

    # 2) 读取 BPG 文件到内存
    with open(temp_bpg, 'rb') as f:
        bpg_data = f.read()  # bytes 对象
    # 转成 torch.uint8
    bpg_data_tensor = torch.tensor(list(bpg_data), dtype=torch.uint8)

    # 3) 把BPG字节流转换成比特 (每字节8bit)
    bpg_bits = dec2bin(bpg_data_tensor, 8)  # shape: [num_bytes*8]

    # 4) 建立 sionna LDPC + 信道模型
    #    - 假设 bpg_bits_tf shape: [1, K], K = bpg_bits_tf.shape[1]
    bpg_bits_tf = tf.convert_to_tensor(bpg_bits.unsqueeze(0).numpy(), dtype=tf.float32)
    blocks, num_bits, num_blocks  = split_into_blocks(bpg_bits_tf, max_k)
    c, k, n = ldpc_encode_blocks(blocks, ldpc_rate)
    x = mapper(c)              # [1, N/4]
    no = sionna.phy.utils.ebnodb2no(ebno_db, 2**nbits, ldpc_rate)

    if channel_type == 'rayleigh':
        # x: [B, T]（mapper 输出先确保是 [B, T]，不要多余的末尾维）
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        # 1) 把时间维展平到 batch： [B*T, 1]（SISO: Nt=1）
        x_flat = tf.reshape(x, [-1, 1])                # [B*T, 1]
        # 2) 过平坦瑞利信道
        #    y: [B*T, 1], h: [B*T, 1, 1]
        y, h = channel(x_flat, no)
        # 3) LMMSE 等化（SISO -> 噪声协方差是 1x1）
        s = tf.cast(no*tf.eye(1), y.dtype)             # [1,1]
        x_hat, no_eff = lmmse_equalizer(y, h, s)       # [B*T, 1]
        # 4) 还原回 [B, T]，送 demapper
        x_hat = tf.reshape(x_hat, [B, T])              # [B, T]
        llr = demapper(x_hat, no)
    if channel_type == 'awgn':
        y = channel(x, no)
        llr = demapper(y, no)
    b_hat = ldpc_decode_blocks(llr, k, n, num_bits, num_blocks)
    # 计算BER（可选）
    # ber_coded = compute_ber(bpg_bits_tf, b_hat)
    # print(f"EbNo={ebno_db}dB, BPG+LDPC BER={ber_coded:.4e}")

    # 6) 重组BPG字节流并写回文件
    b_hat = torch.from_numpy(b_hat.numpy()).to(torch.uint8).squeeze(0)    # shape: [K]
    b_hat_bytes = bin2dec(b_hat.view(-1, 8), 8)                           # [num_bytes]
    # b_hat_bytes 还是 torch.uint8，转成 list 然后变成真正的bytearray
    # final_bpg_data = header_bytes + bytes(b_hat_bytes.tolist())
    final_bpg_data = bytes(b_hat_bytes.numpy().tolist())

    # 需要考虑传输一个符号携带多少比特
    cbr = (bpg_data_tensor.nelement() * (8 / nbits)) / (raw_img_tensor.nelement()) / ldpc_rate
    with open(temp_bpg_received, 'wb') as f:
        f.write(bytearray(final_bpg_data))

    # 7) 调用 bpgdec 解码 -> temp_recon.png
    try:
        bpg_decompress(
            input_bpg=temp_bpg_received,
            output_file=temp_recon_png
        )
        # 8) 读回 temp_recon.png 做 PSNR
        recon_img = Image.open(temp_recon_png).convert('RGB')
        recon_tensor = transforms.ToTensor()(recon_img) * 255  # 回到[0,255]范围

        psnr_val = calculate_psnr(torch.squeeze(raw_img_tensor, 0).float(), 
                                recon_tensor.float())
        ms_ssim = 1 - CalcuSSIM(raw_img_tensor/255., (recon_tensor.unsqueeze(0)/255.).clamp(0., 1.)).mean().item()
        
        return psnr_val, cbr, ms_ssim
    except:
        # bpgdec 解码失败（位错误太多）
        # print("BPG decode failed, returning PSNR=0")
        return 0.0, cbr, 0.0


def bpg_after_channel():
    logging.basicConfig(
        filename='bpg.log',           # 日志文件名
        level=logging.INFO,           # 日志级别
        format='%(asctime)s %(levelname)s: %(message)s')
    for quality in [49, 47, 45, 43, 41]:                           # 49, 47, 45, 43, 41
        for i in range(len(multiple_snr)):
            psnrs, cbrs, ssim = [AverageMeter() for _ in range(3)]
            for j, data in enumerate(tqdm(kodak_loader)):
                img = data
                ebno_db = snrdb2no(multiple_snr[i], 2**nbits, code_rate)
                psnr, cbr, ms_ssim = test_bpg_ldpc_channel_single_image(
                                        img, bpg_quality=quality,          #  35 30 20 15 10
                                        ldpc_rate=code_rate,               # 1/2 1/3 1/5
                                        max_k=max_k,ebno_db=ebno_db,
                                        channel_type=channel_type)
                cbrs.update(cbr)
                psnrs.update(psnr)
                ssim.update(ms_ssim)
            logging.info(f"quality: {quality}")
            logging.info(f"snr: {multiple_snr[i]}")
            logging.info(f"cbrs: {cbrs.avg}")
            logging.info(f"psnrs: {psnrs.avg}")
            logging.info(f"ssim: {ssim.avg}")
            print("quality:", quality)
            print("snr:", multiple_snr[i])
            print("cbrs:", cbrs.avg)
            print("psnrs:", psnrs.avg)
            print("ssim:", ssim.avg)

def test_jpeg_ldpc_channel_single_image(
    raw_img_tensor,
    jpeg_quality=75,
    ldpc_rate=0.5,
    max_k=3840,
    ebno_db=6.0,
    channel_type=channel_type):
    """
    对单张图执行:
      1) JPEG 压缩到字节流
      2) JPEG 字节流 -> LDPC + 信道 -> JPEG 字节流
      3) 解码 JPEG 字节流 -> 图像
      4) 计算 PSNR/CBR
    """
    # 1) JPEG 压缩
    en_img = io.encode_jpeg(torch.squeeze(raw_img_tensor, 0), quality=jpeg_quality)
    jpeg_bytes = torch.tensor(list(en_img), dtype=torch.uint8)

    # 2) 转换为比特流
    jpeg_bits = dec2bin(jpeg_bytes, 8)  # [num_bytes*8]
    jpeg_bits_tf = tf.convert_to_tensor(jpeg_bits.unsqueeze(0).numpy(), dtype=tf.float32)

    # 3) 切分成 LDPC 块
    blocks, num_bits, num_blocks = split_into_blocks(jpeg_bits_tf, max_k)

    # 4) LDPC 编码 + 调制
    c, k, n = ldpc_encode_blocks(blocks, ldpc_rate)
    x = mapper(c)
    no = ebnodb2no(ebno_db, 2**nbits, ldpc_rate)

    if channel_type == 'rayleigh':
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        # 1) [B*T, 1]（SISO: Nt=1）
        x_flat = tf.reshape(x, [-1, 1])                # [B*T, 1]
        # 2) 过衰落信道
        #    y: [B*T, 1], h: [B*T, 1, 1]
        y, h = channel(x_flat, no)
        # 3) LMMSE 均衡 (SISO -> noise 方差是 1x1)
        s = tf.cast(no*tf.eye(1), y.dtype)             # [1,1]
        x_hat, no_eff = lmmse_equalizer(y, h, s)       # [B*T, 1]
        # 4) 还原尺寸到 [B, T] 解调
        x_hat = tf.reshape(x_hat, [B, T])              # [B, T]
        llr = demapper(x_hat, no)
    if channel_type == 'awgn':
        y = channel(x, no)
        llr = demapper(y, no)

    # 5) LDPC 解码
    b_hat = ldpc_decode_blocks(llr, k, n, num_bits, num_blocks)

    # 6) 重组 JPEG 字节流
    b_hat = torch.from_numpy(b_hat.numpy()).to(torch.uint8).squeeze(0)    # shape: [K]
    b_hat_bytes = bin2dec(b_hat.view(-1, 8), 8)                           # [num_bytes]
    final_jpeg_data = bytes(b_hat_bytes.numpy().tolist())

    # 7) 解码 JPEG
    try:
        de_img = io.decode_jpeg(torch.tensor(list(final_jpeg_data), dtype=torch.uint8))
        psnr_val = calculate_psnr(torch.squeeze(raw_img_tensor, 0).float(), de_img.float())
        ms_ssim = 1 - CalcuSSIM(raw_img_tensor/255., 
                                (de_img.unsqueeze(0)/255.).clamp(0., 1.)).mean().item()
    except:
        # JPEG 解码失败（位错误太多）
        de_img = torch.zeros_like(torch.squeeze(raw_img_tensor, 0))
        psnr_val = 0.0
        ms_ssim = 0.0

    # 8) 计算 CBR
    cbr = (jpeg_bytes.nelement() * (8 / nbits)) / (raw_img_tensor.nelement()) / ldpc_rate

    return psnr_val, cbr, ms_ssim

def jpeg_after_channel():
    logging.basicConfig(
        filename='jpeg.log',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s')
    logging.info(f"channel: {channel_type}")
    for quality in [1,2,3,4,5,6]:              # set of JPEG quality factors
        for i in range(len(multiple_snr)):
            psnrs, cbrs, ssims = [AverageMeter() for _ in range(3)]
            for j, data in enumerate(tqdm(kodak_loader)):
                img = data
                ebno_db = snrdb2no(multiple_snr[i], 2**nbits, code_rate)
                psnr, cbr, ms_ssim = test_jpeg_ldpc_channel_single_image(
                                img, jpeg_quality=quality,ldpc_rate=code_rate,
                                max_k=max_k, ebno_db=ebno_db,
                                channel_type=channel_type)
                cbrs.update(cbr)
                psnrs.update(psnr)
                ssims.update(ms_ssim)
            logging.info(f"quality: {quality}")
            logging.info(f"snr: {multiple_snr[i]}")
            logging.info(f"cbrs: {cbrs.avg}")
            logging.info(f"psnrs: {psnrs.avg}")
            logging.info(f"ssim: {ssims.avg}")
            print("quality:", quality)
            print("snr:", multiple_snr[i])
            print("cbrs:", cbrs.avg)
            print("psnrs:", psnrs.avg)
            print("ssim:", ssims.avg)

def cifar_no_channel():
    for i in range(30, 100, 10):
        psnrs, cbrs = [AverageMeter() for _ in range(2)]
        for j, data in enumerate(tqdm(test_loader_all)):
            raw_img, _ = data
            raw_img = torch.squeeze(raw_img, 0)
            # JPEG encode
            en_img = io.encode_jpeg(raw_img, i)

            de_img = io.decode_jpeg(en_img)
            cbrs.update(en_img.nbytes / raw_img.nbytes / 0.5)
            psnrs.update(calculate_psnr(raw_img, de_img))
        print("quality:", i)
        print("cbrs:", cbrs.avg)
        print("psnrs:", psnrs.avg)
        cbrs.clear()
        psnrs.clear()


def kodak_no_channel():
    for i in range(1, 100, 10):
        psnrs, cbrs = [AverageMeter() for _ in range(2)]
        for j, data in enumerate(tqdm(kodak_loader)):
            raw_img = data
            raw_img = torch.squeeze(raw_img, 0)
            # JPEG encode
            en_img = io.encode_jpeg(raw_img, i)
            de_img = io.decode_jpeg(en_img)
            cbrs.update(en_img.nbytes / raw_img.nbytes / 0.5)
            psnrs.update(calculate_psnr(raw_img, de_img))
            save_image(raw_img, de_img, 'test')
        
        print("snr:", i)
        print("cbrs:", cbrs.avg)
        print("psnrs:", psnrs.avg)

def test_channel():
    # system parameters
    n_ldpc = 500 # LDPC codeword length
    k_ldpc = 250 # number of info bits per LDPC codeword
    coderate = k_ldpc / n_ldpc
    num_bits_per_symbol = 2 # number of bits mapped to one symbol (cf. QAM)
    demapping_method = "app" # try "max-log"
    ldpc_cn_type = "boxplus" # try also "minsum" 
    binary_source = BinarySource()
    encoder = LDPC5GEncoder(k_ldpc, n_ldpc)
    constellation = Constellation("qam", num_bits_per_symbol)
    mapper = Mapper(constellation=constellation)
    channel = AWGN()
    demapper = Demapper(demapping_method,
                        constellation=constellation)
    decoder = LDPC5GDecoder(encoder,hard_out=True,
                            cn_type=ldpc_cn_type,
                            num_iter=20)
    # simulation parameters
    batch_size = 1000
    snr_db = 7

    ebno_db = snrdb2no(snr_db, num_bits_per_symbol, coderate)

    # Generate a batch of random bit vectors
    b = binary_source([batch_size, k_ldpc])

    # Encode the bits using 5G LDPC code
    print("Shape before encoding: ", b.shape)
    c = encoder(b)
    print("Shape after encoding: ", c.shape)

    # Map bits to constellation symbols
    x = mapper(c)
    print("Shape after mapping: ", x.shape)

    # Transmit over an AWGN channel at SNR 'ebno_db'
    no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
    y = channel(x, no)
    print("Shape after channel: ", y.shape)

    # Demap to LLRs
    llr = demapper(y, no)
    print("Shape after demapping: ", llr.shape)

    # LDPC decoding using 20 BP iterations
    b_hat = decoder(llr)
    print("Shape after decoding: ", b_hat.shape)

    # calculate BERs
    c_hat = tf.cast(tf.less(0.0, llr), tf.float32) # hard-decided bits before dec.
    ber_uncoded = compute_ber(c, c_hat)

    ber_coded = compute_ber(b, b_hat)

    print("BER uncoded = {:.3f} at EbNo = {:.1f} dB".format(ber_uncoded, ebno_db))
    print("BER after decoding = {:.3f} at EbNo = {:.1f} dB".format(ber_coded, ebno_db))
    print("In total {} bits were simulated".format(np.size(b.numpy())))


if __name__ == "__main__":
    # test_channel()
    bpg_after_channel()
    # jpeg_after_channel()
