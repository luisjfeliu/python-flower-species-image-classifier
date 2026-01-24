# Flower Species Image Classifier
This project finetunes a PyTorch torchvision model for classifying flower species images.

# Requirements

This repository uses Git LFS.

Before cloning make sure to run the following if you wish to download the saved model:
```
git lfs install

```

# Setup
```
conda env create -f environment.yml 
conda activate flower-classifier
```
or 
```
pip install -r requirements.txt 
```
# Notebook
[Image Classifier Project](./Image%20Classifier%20Project.ipynb)
# CLI

`main.py` is the main entry point. See examples below.

## Train
```
python main.py train flowers --arch resnet34 --learning_rate 0.01 --hidden_units 256 --epochs 20 --gpu
```

## Predict
```
python main.py predict "flowers/test/99/image_07833.jpg" checkpoint.pth --top_k 3 --category_names cat_to_name.json --gpu
```

### Sample CLI outputs
#### Train
```
Downloading: "https://download.pytorch.org/models/resnet34-b627a593.pth" to /root/.cache/torch/hub/checkpoints/resnet34-b627a593.pth
100%|███████████████████████████████████████████████████████████████| 83.3M/83.3M [00:00<00:00, 259MB/s]
Epoch 1/20.. Train loss: 4.197.. Validation loss: 3.420.. Validation accuracy: 0.174
Epoch 2/20.. Train loss: 3.627.. Validation loss: 3.084.. Validation accuracy: 0.214
Epoch 3/20.. Train loss: 3.418.. Validation loss: 2.690.. Validation accuracy: 0.312
Epoch 4/20.. Train loss: 3.360.. Validation loss: 2.598.. Validation accuracy: 0.315
Epoch 5/20.. Train loss: 3.254.. Validation loss: 2.384.. Validation accuracy: 0.376
Epoch 6/20.. Train loss: 3.136.. Validation loss: 2.375.. Validation accuracy: 0.391
Epoch 7/20.. Train loss: 3.185.. Validation loss: 2.469.. Validation accuracy: 0.367
Epoch 8/20.. Train loss: 3.129.. Validation loss: 2.376.. Validation accuracy: 0.384
Epoch 9/20.. Train loss: 3.109.. Validation loss: 2.247.. Validation accuracy: 0.403
Epoch 10/20.. Train loss: 3.099.. Validation loss: 2.445.. Validation accuracy: 0.394
Epoch 11/20.. Train loss: 3.225.. Validation loss: 2.356.. Validation accuracy: 0.396
Epoch 12/20.. Train loss: 3.038.. Validation loss: 2.323.. Validation accuracy: 0.359
Epoch 13/20.. Train loss: 3.054.. Validation loss: 2.264.. Validation accuracy: 0.401
Epoch 14/20.. Train loss: 3.038.. Validation loss: 2.235.. Validation accuracy: 0.395
Epoch 15/20.. Train loss: 3.039.. Validation loss: 2.301.. Validation accuracy: 0.396
Epoch 16/20.. Train loss: 3.063.. Validation loss: 2.281.. Validation accuracy: 0.389
Epoch 17/20.. Train loss: 3.081.. Validation loss: 2.414.. Validation accuracy: 0.350
Epoch 18/20.. Train loss: 3.046.. Validation loss: 2.164.. Validation accuracy: 0.449
Epoch 19/20.. Train loss: 3.047.. Validation loss: 2.257.. Validation accuracy: 0.407
Epoch 20/20.. Train loss: 3.028.. Validation loss: 2.381.. Validation accuracy: 0.386
Checkpoint saved to ./checkpoint.pth
```
#### Predict
```
frangipani: 0.3664
water lily: 0.1695
lotus lotus: 0.1380
```