# BPG / JPEG + LDPC 图像传输性能测试

使用 [BPG](https://bellard.org/bpg/) 和 [Sionna](https://github.com/NVlabs/sionna) 实现图像压缩与传输的性能测试



## requirements
https://bellard.org/bpg/

```shell
matplotlib==3.10.8
numpy==2.4.4
Pillow==12.1.1
sionna==2.0.0
sionna_rt==2.0.0
torch==2.10.0+cu130
torchvision==0.25.0+cu130
tqdm==4.67.3

```



## Usage

```shell
# 设置好仿真参数后运行 ldpc.py
# test_loader 跑CIFAR10的结果  kodak_loader跑Kodak数据集
# 可以设置SNR 与 压缩质量 一次
python ldpc.py
```



## Citation

If this work is useful for your research, please cite:

```shell
@article{li2025mvsc,
  title={MVSC: Mamba Vision based Semantic Communication for Image Transmission with SNR Estimation},
  author={Li, Chongyang and Zhang, Tianqian and Liu, Shouyin},
  journal={IEEE Communications Letters},
  year={2025},
  publisher={IEEE}
}
```

