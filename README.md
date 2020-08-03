# Clog Loss: Advance Alzheimer’s Research with Stall Catchers 

[Clog Loss: Advance Alzheimer’s Research with Stall Catchers](https://www.drivendata.org/competitions/65/clog-loss-alzheimers-research/page/207/).

[2nd place](https://www.drivendata.org/competitions/65/clog-loss-alzheimers-research/leaderboard/)
out of 922 with 0.8389 Matthew's correlation coefficient (MCC) (top 1 -- 0.8555).

### Prerequisites

- GPU with 32Gb RAM (e.g. Tesla V100)
- [NVIDIA apex](https://github.com/NVIDIA/apex)

```bash
pip install -r requirements.txt
```

### Usage

#### Download

First download the train and test data from the competition link into `/home/data/clog-loss` folder.

Then you must prepare train and test datasets.

```bash
sh ./preprocess.sh
```

This will download whole dataset (1.4Tb), crop the region of interest from video using code provided by `@Moshel` on the [forum](https://community.drivendata.org/t/python-code-to-find-the-roi/4499) 
and save it in [lmdb](https://github.com/jnwatson/py-lmdb) format for fast reading of 3D arrays. Note, it will take around
1 day and consume 200Gb of RAM, hence disk. So, if you have not enough RAM you can easily rewrite the code to process
the data by chunks.

#### Train

To train the model run

```bash
sh ./train.sh
```

On 1 GPU Tesla V100 it will take around 1 week.

#### Test

To make inference on test dataset run

```bash
sh ./test.sh
```

On 1 GPU Tesla V100 it takes around 40m.

You can also download the trained models from [yandex disk](https://yadi.sk/d/2GGRsM-ac5CaKQ),
unzip and run

```bash
sh ./predict.py
```

Last two commands will generate submission file.

### Approach

3D Convolutional network on full tier 1 data. 

#### First observations

Interestingly, height and width of crops have positive signal. Simple gradient boosting on these 2 features gives 0.16
local validation. I only tried to add them into video features, but the score was getting worse on LB. So I discarded
this idea on early stage of model development, which probably cost me the 1st place :) Baseline model based on 2D CNN as features extractors for video frames and LSTM
as classifier reaches 0.59 LB. Single 3D ResNet34 on `Micro` dataset already gives 0.68 on LB. With 2 folds and TTAs one
can reach 0.76. Heavier models like 3D ResNet50 and ResNet101 has 0.71 and 0.76 respectively. On full tier 1 dataset
single 3D ResNet34 gives 0.81.

#### Summary

- 160x160xF resized crops of ROIs of full tier 1 dataset, where F is a video depth
- 3D ResNet101
- Binary Cross Entropy
- Batch size 4
- AdamW with `1e-4` learning rate
- CosineAnealing scheduler
- Augmentations, like horizontal and vertical flips, rotate on 90, distortions, noise
- Mean of 5 predictions of 5 different snapshots of the same model 3D ResNet101

#### Tried

- 2D CNN + LSTM (0.59 LB)
- 2D CNN + Transformer is not better than LSTM
- 2D CNN + 1D CNN is not better than LSTM
- 3D CNN Efficientnet doesn't train
- Test time augmentations doesn't improve score
- Focal and Lovasz losses are not better than BCE
- Crowd score instead binary label in loss, but the results are the same
- No improvements using tier 2 dataset with crowd score > 0.6 
- 2nd level model on out of fold predictions with additional features like size of crop worsen local validation and public LB
(drama at the end)
- 1 round of pseudo-labeling (not enough investigated)
- [AdaTune](https://github.com/awslabs/adatune) works same as `lr=1e-4`

#### Possible improvements

- As it turns out 2nd level model highly improves private score, simply using extra features like width and height of
crop you can get +2 points on private. But it is useless in production :)
