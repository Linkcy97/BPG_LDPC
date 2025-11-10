# BPG / JPEG + LDPC 图像传输性能测试

使用 [BPG]([BPG Image format](https://bellard.org/bpg/)) 和 [Sionna]([NVlabs/sionna: Sionna: An Open-Source Library for Research on Communication Systems](https://github.com/NVlabs/sionna)) 实现图像压缩与传输的性能测试



## requirements

```shell
matplotlib==3.10.7
numpy==2.3.4
Pillow==12.0.0
sionna==1.2.1
sionna_rt==1.2.1
tensorflow==2.14.0
torch==2.1.1+cu118
torchvision==0.16.1+cu118
tqdm==4.66.4
```



## Usage

```shell
# setting the communication system parament and run ldpc.py
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

