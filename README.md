
## Get Started


1. Install Python 3.9, PyTorch 1.10.0.
The code is built based on Python 3.9, PyTorch 1.13.0.
You can install PyTorch following the instruction in [PyTorch](https://pytorch.org/get-started/locally/). For example:

```bash
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
```

After ensuring that PyTorch is installed correctly, you can install other dependencies via:

```bash
pip install -r requirements.txt
```

2. Download data. You can obtain all the eight benchmarks from [baidupan](https://pan.baidu.com/s/19lqv1VLG9VBx7Nh04L1u0A?pwd=mswn 
)password：mswn. **All the datasets are well pre-processed** and can be used easily.
3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the experiment results by:
```bash
bash ./scripts/LSTF_Forcasting.sh
```
## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/zhouhaoyi/Informer2020

https://github.com/zhouhaoyi/ETDataset

https://github.com/laiguokun/multivariate-time-series-data
