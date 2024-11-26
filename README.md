# Learning Robust Anymodal Segmentor with Unimodal and Cross-modal Distillation
<img width="1239" alt="image" src="https://github.com/user-attachments/assets/a308e214-598f-4980-92ff-32e985df55a2">

## Update
[11/2024], pre-trained weights and evalutation code on MUSES are released.

## Environments
```
git clone --
cd --
conda create -n anyseg python=3.9
conda activate anyseg
pip install -r requirements.txt
```
## Data Preparation
Used Datasets: 
[MUSES](https://muses.vision.ee.ethz.ch/) / [DELIVER](https://github.com/jamycheung/DELIVER)

## Pre-trained Weights of AnySeg

| Method  | F      | E      | L      | FE     | FL     | EL     | FEL    | Mean   | Weights                                                                                      |
|---------|--------|--------|--------|--------|--------|--------|--------|--------|---------------------------------------------------------------------------------------------|
| CMX     | 2.52   | 2.35   | 3.01   | 41.15  | 41.25  | 2.56   | 42.27  | 19.30  | -                                                                                           |
| CMNeXt  | 3.50   | 2.77   | 2.64   | 6.63   | 10.28  | 3.14   | 46.66  | 10.80  | -                                                                                           |
| MAGIC   | 43.22  | 2.68   | 22.95  | 43.51  | 49.05  | 22.98  | 49.02  | 33.34  | -                                                                                           |
| Any2Seg | 44.40  | 3.17   | 22.33  | 44.51  | 49.96  | 22.63  | 50.00  | 33.86  | -                                                                                           |
| Ours    | 46.01  | 19.57  | 32.13  | 46.29  | 51.25  | 35.21  | 51.14  | 40.23  | [model](https://drive.google.com/file/d/17pmkR_xdCKdn0LPwaf27S7URjjI1HeMS/view?usp=sharing) |

## References
We appreciate the previous open-source works: [DELIVER]([https://github.com/jamycheung/Trans4PASS](https://github.com/jamycheung/DELIVER)) / [SegFormer](https://github.com/NVlabs/SegFormer) / [MUSES](https://muses.vision.ee.ethz.ch/)
