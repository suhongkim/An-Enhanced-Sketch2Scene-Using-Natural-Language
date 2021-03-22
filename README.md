# An Enhanced Sketch2Scene Using Natural Language

## Motivation
Natural Language Processing and Computer vision are both the most actively developing machine learning
research areas. Integrating these two fields has received a lot of attention recently. In this project, we would like
to combine computer vision and Natural language Processing to transform an incomplete sketch into a scene.
Translating a sketch into a scene is a challenging computer vision task. Realistic and meaningful scenes should
contain multiple objects as well as a corresponding background, which is hard to extrapolate from a single input
object. We propose a new approach to enhance the Sketch2Scene translation task using natural language
processing techniques. Instead of directly transforming an image into another, we transform the sketch to a word
and then use the word as a condition to create longer sentences and then use it to generate a scene. The first
step is to classify the input sketch using a CNN based classifier. Once we have the class of the sketch, we find
the most similar words related to that class. We generate a natural language scene description, conditioned on
the class, and its similar words. Then we use a separate system that takes in the scene description and
generates an image. Below is our scope of the project.
- Implement a new pipeline to generate realistic scenes from a single sketch
- Show how NLP can improve the Sketch2Scene task with respect to quality and diversity

## Approach 
The pipeline in this project contains three main steps to take in an incomplete sketch and transform it into a
complete scene. For the first step, we have a pre-trained MobileNetV2 (https://arxiv.org/abs/1801.04381) sketch
classifier on ImageNet and fine-tuned it on 340 different object classes from the Google QuickDraw dataset
(https://www.kaggle.com/c/quickdraw-doodle-recognition). Given the input sketch, the output class of the
classifier is used for the next step, which is the caption generator, to find the similar words and feed them into
an encoder-decoder based sequential model to encode those words and use them to generate the scene
captions. In the last step we infer the scenes using a pre-trained Caption2Scene Generator model which learns
to retrieve objects and arrange them in the scene using the semantic relationship of the objects in the captions.
Our main contribution is to experiment with two different baselines for this task:

- Baseline 1 - Get noun object from COCO captions and use it as the condition to generate captions
- Baseline 2 - Get similar words of the predicted class of the sketch, using word2Vec embedding and use as
the condition to generate captions.

Baseline 1 is used a sanity check to validate the conditional sequence genrative model which will be discussed
in section 2.2 . And, for the second baseline we will use diferent methods like retorfitting , beam search and train
time data augmentations to improve the generated captions.

![](images/diagram2.png)

## Results
![](images/results.PNG)
You will find our results in [the short verion](EnhancedSketch2SceneUsingNaturalLanguage_post.pdf] and [the detailed report](EnhancedSketch2SceneUsingNaturalLanguage_report.pdf]

![](images/evaluation.png)
Our method produces high quality (High BLEU4 score) and more diverse (Lower Self-BLEU4score) captions given sketch input. Our
method generates captions which have better correspondence with input sketch (HigherSemantic Accuracy)

![](images/scene_evaluation.png)
Our method generates better captions resulting in better scenes (higher Inception score).
Caption2Scene model generates the scenes with more diverse objects and relationships,
which are unseen by YOLOv3 detector (Lower SOA score)


## How to Run 
Check each README.md file in three modules: [CaptionGenerator](CaptionGenerator/README.md), [Sketch_classifier](Sketch_classifier/README.md), and [Text2Scene](Text2Scene/README.md)

## Authors
This project is done by team of three for each module
- CaptionGenerator - [Suhong kim](https://www.linkedin.com/in/suhongkim/)
- Sktech Classifier - [Sara Jalili](https://www.linkedin.com/in/sara-jalili/)
- Text2Scene - [Vishnu Sanjay Ramiya Srinivasan](https://www.linkedin.com/in/vishnu-sanjay-rs/)

## References
[1]  F. Tan, S. Feng, and V. Ordonez, “Text2scene:  Generating compositional scenes from textualdescriptions,” 2018. \
[2]  “COCO Datset.”http://cocodataset.org/#home. \
[3]  “Quick,   Draw!Doodle   Recognition   Challenge.”https://www.kaggle.com/c/quickdraw-doodle-recognitio 
