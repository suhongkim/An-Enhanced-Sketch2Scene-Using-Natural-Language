# Caption Generation

## Overview
Based on PyTorch Image Captiioning Tutorial code, we discarded the image encoder and attention module, and then added our keywords embeddings that are obtained from the object name of user sketch. We trained the decoder based on the keywords-caption pairs which we constructed from COCO caption dataset. The main challenge of this part is how to obtain better quality and more diverse captions conditioned on keywords. To takle this problem, we defined the baseline model and built up our model using some natural language technique that we have learned from the CMPT825 course. Also, we evaluated our outputs compared to the baseline using three metrics: BLEU4 score, Self-BLEU4 score, and Semantic Accuracy. 

## Installation
- Setup a venv environment and install some prerequisite packages like this
```bash
python3 -m venv venv
source venv/bin/activate 
pip3 install -r requirements.txt  # Install dependencies
```
## Data 
Please download [glove.42B.300d file](https://nlp.stanford.edu/data/glove.42B.300d.zip) and unzip it under "data" folder.  run the below code to generate all the magnitude files. ** It takes more than 30 min ** 
```bash
python3 retrofitting.py 
```

Please download [data file](https://drive.google.com/file/d/139Sh4_7zeHqznluVFiTElFG5VdsisiTd/view?usp=sharing) and unzip it under "data" folder. This folder contains the vocabluary wordmap and the test inputs files. 

## Pretrained model 
Please download all [pretrained models](https://drive.google.com/file/d/1TgwRn0aB0b0W-s3sAy7JCproyEaSyHyu/view?usp=sharing) and unzip it under "pretrained" folder 


### Please run the project.ipynb to train/validate/evaluation [Recomended]

## Demo to generate captions from Doodle Class Name (Randomly Selected)
```bash
python3 inference.py
```

## Evaluations for all doodle class names
```bash
python3 evaluation.py
```

## Training
You can edit the parameters at train.py 
```bash
python3 train.py
```

## Report requirements

Please download all images for the report from here [report images](https://drive.google.com/open?id=1Kuo4Ef7DdTZSX4MqHwTob9FQqSYdY-Jo) and extract it as "images" folder .





