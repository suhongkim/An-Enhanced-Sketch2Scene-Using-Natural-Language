from PIL import Image
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3
import numpy as np
from scipy.stats import entropy
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse
import os
import pickle as pkl
from tqdm import tqdm
import glob
import shutil

from darknet import Darknet
from dataset import YoloDataset
from util import *
import nltk
from nltk.corpus import wordnet
import json


def cal_soa_accuracy(caption_path, images_path, yolo_pred_dict):
    images = []
    for file in sorted(os.listdir(images_path)):
        if file.startswith("final_") and file.endswith(".png"):
            images.append(file)

    captions = json.loads(open(caption_path, "r").read())

    classes = load_classes('coco.names')

    detected_counts = 0
    total_counts = 0
    for i in range(len(images)):
        cap = captions[i]
        img_name = images[i]
        pred = yolo_pred_dict[img_name]
        pred_classes = pred[0]

        caption_text = cap.lower().strip().split(" ")
        caption_tags = nltk.pos_tag(caption_text)
        nn_words = [t[0] for t in nltk.pos_tag(caption_text) if t[1] == 'NN']
        hypernym_list = []
        hypernym_list = hypernym_list + nn_words
        for nn in nn_words:
            synset = wordnet.synsets(nn)
            for syns in synset:
                syn_word = wordnet.synset(syns.name())
                hypernym_list = hypernym_list + list(
                    set([w for s in syn_word.closure(lambda s: s.hypernyms()) for w in s.lemma_names()]))
        hypernym_set = set(hypernym_list)
        class_set = set(classes)
        inte = hypernym_set.intersection(class_set)

        total_counts = total_counts + len(inte)
        detected = inte.intersection(pred_classes)

        detected_counts = detected_counts + len(detected)

    return detected_counts / total_counts


def yolo_predictions(images):
    batch_size = 1
    confidence = 0.5
    nms_thresh = 0.4
    img_size = 256

    classes = load_classes('coco.names')

    CUDA = torch.cuda.is_available()

    # Set up the neural network
    print("Loading network.....")
    model = Darknet('./yolov3.cfg')
    model.load_weights('yolov3.weights')
    print("Network successfully loaded")

    model.net_info["height"] = 256
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # If there's a GPU available, put the model on GPU
    if CUDA:
        _gpu = 0
        torch.cuda.set_device(_gpu)
        model.cuda()
        print("Using GPU: {}".format(_gpu))

    # Set the model in evaluation mode
    model.eval()

    # create dataset from images in the current folder
    image_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0., 0., 0.), (1, 1, 1))])
    dataset = YoloDataset(images, transform=image_transform)
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             drop_last=False, shuffle=False, num_workers=4)

    num_batches = len(dataloader)
    dataloader = iter(dataloader)
    output_dict = {}

    # get YOLO predictions for images in current folder
    for idx in tqdm(range(num_batches)):
        data = dataloader.next()
        imgs, filenames = data
        #     print(filenames)
        if CUDA:
            imgs = imgs.cuda()

        with torch.no_grad():
            predictions = model(imgs, CUDA)
            predictions = non_max_suppression(predictions, confidence, nms_thresh)

        for img, preds in zip(filenames, predictions):
            img_preds_name = []
            img_preds_id = []
            img_bboxs = []
            if preds is not None and len(preds) > 0:
                for pred in preds:
                    pred_id = int(pred[-1])
                    pred_name = classes[pred_id]

                    bbox_x = pred[0] / img_size
                    bbox_y = pred[1] / img_size
                    bbox_width = (pred[2] - pred[0]) / img_size
                    bbox_height = (pred[3] - pred[1]) / img_size

                    img_preds_id.append(pred_id)
                    img_preds_name.append(pred_name)
                    img_bboxs.append([bbox_x.cpu().numpy(), bbox_y.cpu().numpy(),
                                      bbox_width.cpu().numpy(), bbox_height.cpu().numpy()])
            output_dict[img.split("/")[-1]] = [img_preds_name, img_preds_id, img_bboxs]

    return output_dict


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores)


class Basic_Image_Dataset_2D(torch.utils.data.Dataset):
    def __init__(self, path):
        self.dataset_path = path
        self.img_path = []
        for file in os.listdir(self.dataset_path):
            if file.startswith("final_") and file.endswith(".png"):
                self.img_path.append(os.path.join(self.dataset_path, file))

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):

        img = Image.open(self.img_path[idx], 'r')
        img = img.convert('RGB')
        img_arr = np.array(img)
        #     print(img_arr.shape)
        img_arr = (img_arr - 128.) / 128.
        #     print(np.max(img_arr),np.min(img_arr))
        img_arr = np.transpose(img_arr, (2, 0, 1))
        img_tensor = torch.tensor(img_arr)
        return img_tensor


def evaluate_inception(images_path):
    images = []
    for file in sorted(os.listdir(images_path)):
        if file.startswith("final_") and file.endswith(".png"):
            images.append(os.path.join(images_path, file))

    dataset = Basic_Image_Dataset_2D(path=images_path)
    inc_score = inception_score(dataset, batch_size=16)

    return inc_score


print("Inception Score evaluation for image quality")
print("=============================================")

print("Evaluating Images generated using caption generated by the baseline method")
print("==========================================================================")
inc_score = evaluate_inception(images_path = "./logs/baseline/composites_samples")
print("Inception Score for Baseline images ",inc_score)
print("=============================================")

print("Evaluating Images generated using caption generated by our method")
print("=================================================================")
inc_score = evaluate_inception(images_path = "./logs/retrofit/composites_samples")
print("Inception Score for Retrofitted images ",inc_score)

print("Semantic Object Accuracy (SOA) evaluation for caption to ima ge accuracy")
print("========================================================================")
print("Evaluating Images generated using captions generated by the baseline method")
print("==========================================================================")
pred_dict = yolo_predictions(images = "./logs/baseline/composites_samples")
soa_accuracy = cal_soa_accuracy(images_path="./logs/baseline/composites_samples",caption_path='./examples/composites_samples_baseline.json',yolo_pred_dict=pred_dict)
print("SOA Score for Baseline images ",soa_accuracy)

print("Evaluating Images generated using captions generated by the retrofitted method")
print("==========================================================================")
pred_dict_retro = yolo_predictions(images="./logs/retrofit/composites_samples")
soa_accuracy = cal_soa_accuracy(images_path="./logs/retrofit/composites_samples",
                            caption_path='./examples/composites_samples_retorfit.json', yolo_pred_dict=pred_dict_retro)
print("SOA Score for Baseline images ", soa_accuracy)
