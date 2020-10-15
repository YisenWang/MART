# MART
Code for ICLR2020 "Improving Adversarial Robustness Requires Revisiting Misclassified Examples"


## Usage
python3 train_resnet.py for ResNet18

python3 train_wideresnet.py for WideResNet

## Trained Models
The ResNet18 trained by MART on CIFAR-10: https://drive.google.com/file/d/1YAKnAhUAiv8UFHnZfj2OIHWHpw_HU0Ig/view?usp=sharing

The WideResNet-34-10 trained by MART on CIFAR-10: https://drive.google.com/open?id=1QjEwSskuq7yq86kRKNv6tkn9I16cEBjc

MART WideResNet-28-10 model on 500K unlabeled data: https://drive.google.com/file/d/11pFwGmLfbLHB4EvccFcyHKvGb3fBy_VY/view?usp=sharing

## Citing this work
If you use this code in your work, please cite the accompanying paper:

```
@inproceedings{Wang2020Improving,
    title={Improving Adversarial Robustness Requires Revisiting Misclassified Examples},
    author={Yisen Wang and Difan Zou and Jinfeng Yi and James Bailey and Xingjun Ma and Quanquan Gu},
    booktitle={ICLR},
    year={2020}
}
```

## Requirements
- Python 3.7.4 
- Pytorch 1.3.1
- Part of the code is based on the following repo:
  - Dynamic: https://github.com/YisenWang/dynamic_adv_training
  - TREADES: https://github.com/yaodongyu/TRADES
  - RST: https://github.com/yaircarmon/semisup-adv
