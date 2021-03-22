# Kaggle-QuickDraw
PyTorch implementation for [Kaggle Quick, Draw! Doodle Recognition Challenge](https://www.kaggle.com/c/quickdraw-doodle-recognition)


## Training
* Model
    - MobileNetV2 [1] (light-weight, pretrained on ImageNet)
* Augmentation
    - Random horizontally flip
    - Random affine transformation (rotation, translation, scale)
* Batch size: 64
* Optimizer: Adam
* Learning rate: 1e-3
* Weight decay: 1e-4
* Loss function: cross-entropy
* Using 300/class for validation, the remainings are training data
* Final prediction: a single fold model with TTA (horizontal flip)


## Requirements
* pytorch 
* torchvision 
* numpy
* opencv
* pandas
* tqdm

`pip install -r requirements.txt`


## Usage

### Data
* Download data from [Kaggle QuickDraw competition](https://www.kaggle.com/c/quickdraw-doodle-recognition/data)
* Extract train_simplified.zip
* Modify the path appropriately in `config.json`
* Run `PYTHONPATH=. python loaders/quickdraw_loader.py` first to generate training/validation pickle files

### To train/test the model
`python [train, test].py -h` for more details


## Reference
[1] [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
