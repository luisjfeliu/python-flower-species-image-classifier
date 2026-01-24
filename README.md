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
#### Train out with the following parameters
```
python main.py train flowers --arch resnet34 --learning_rate 0.001 --hidden_units 576 --epochs 25 --gpu
```
```
2026-01-24T02:22:04.283116824Z   Downloading: "https://download.pytorch.org/models/resnet34-b627a593.pth" to /root/.cache/torch/hub/checkpoints/resnet34-b627a593.pth
2026-01-24T02:22:58.883110243Z   Epoch 1/25.. Train loss: 3.783.. Validation loss: 2.210.. Validation accuracy: 0.455
2026-01-24T02:23:49.883423339Z   Epoch 2/25.. Train loss: 2.086.. Validation loss: 1.166.. Validation accuracy: 0.692
2026-01-24T02:24:41.282771841Z   Epoch 3/25.. Train loss: 1.500.. Validation loss: 0.803.. Validation accuracy: 0.795
2026-01-24T02:25:32.282820870Z   Epoch 4/25.. Train loss: 1.265.. Validation loss: 0.680.. Validation accuracy: 0.815
2026-01-24T02:26:24.082736436Z   Epoch 5/25.. Train loss: 1.137.. Validation loss: 0.606.. Validation accuracy: 0.829
2026-01-24T02:27:15.682946763Z   Epoch 6/25.. Train loss: 1.026.. Validation loss: 0.526.. Validation accuracy: 0.850
2026-01-24T02:28:07.483268643Z   Epoch 7/25.. Train loss: 0.972.. Validation loss: 0.495.. Validation accuracy: 0.869
2026-01-24T02:28:59.082868831Z   Epoch 8/25.. Train loss: 0.901.. Validation loss: 0.464.. Validation accuracy: 0.878
2026-01-24T02:29:50.682784722Z   Epoch 9/25.. Train loss: 0.870.. Validation loss: 0.428.. Validation accuracy: 0.886
2026-01-24T02:30:42.083411142Z   Epoch 10/25.. Train loss: 0.845.. Validation loss: 0.389.. Validation accuracy: 0.898
2026-01-24T02:31:33.282989219Z   Epoch 11/25.. Train loss: 0.806.. Validation loss: 0.399.. Validation accuracy: 0.893
2026-01-24T02:32:24.482402565Z   Epoch 12/25.. Train loss: 0.790.. Validation loss: 0.369.. Validation accuracy: 0.909
2026-01-24T02:33:15.883235965Z   Epoch 13/25.. Train loss: 0.739.. Validation loss: 0.356.. Validation accuracy: 0.909
2026-01-24T02:34:07.082507385Z   Epoch 14/25.. Train loss: 0.761.. Validation loss: 0.371.. Validation accuracy: 0.908
2026-01-24T02:34:58.082808423Z   Epoch 15/25.. Train loss: 0.738.. Validation loss: 0.361.. Validation accuracy: 0.909
2026-01-24T02:35:49.082480609Z   Epoch 16/25.. Train loss: 0.690.. Validation loss: 0.383.. Validation accuracy: 0.894
2026-01-24T02:36:40.082522125Z   Epoch 17/25.. Train loss: 0.671.. Validation loss: 0.344.. Validation accuracy: 0.904
2026-01-24T02:37:30.482504206Z   Epoch 18/25.. Train loss: 0.646.. Validation loss: 0.361.. Validation accuracy: 0.906
2026-01-24T02:38:21.282582755Z   Epoch 19/25.. Train loss: 0.666.. Validation loss: 0.377.. Validation accuracy: 0.895
2026-01-24T02:39:12.282994860Z   Epoch 20/25.. Train loss: 0.643.. Validation loss: 0.322.. Validation accuracy: 0.924
2026-01-24T02:40:02.885197915Z   Epoch 21/25.. Train loss: 0.634.. Validation loss: 0.320.. Validation accuracy: 0.921
2026-01-24T02:40:54.082387809Z   Epoch 22/25.. Train loss: 0.620.. Validation loss: 0.339.. Validation accuracy: 0.910
2026-01-24T02:41:44.882894884Z   Epoch 23/25.. Train loss: 0.643.. Validation loss: 0.321.. Validation accuracy: 0.923
2026-01-24T02:42:36.083178815Z   Epoch 24/25.. Train loss: 0.583.. Validation loss: 0.365.. Validation accuracy: 0.923
2026-01-24T02:43:27.282541496Z   Epoch 25/25.. Train loss: 0.612.. Validation loss: 0.335.. Validation accuracy: 0.907
2026-01-24T02:43:27.482528028Z   Checkpoint saved to ./checkpoint.pth
```