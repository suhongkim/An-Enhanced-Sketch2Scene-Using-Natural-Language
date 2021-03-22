import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import random

class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, keyword_size, split):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        self.keyword_size = keyword_size
        # # Captions per image
        self.cpi =  5 # self.h.attrs['captions_per_image']

        if 'augment' in data_name.split('_'):
            data_name = data_name[:-1] + str(keyword_size+2)
            # print(data_name)
        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_KEYWORDS_' + data_name + '.json'), 'r') as j:
            self.keywords = json.load(j)

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name +'.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name +'.json'), 'r') as j:
            self.caplens = json.load(j)

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        if len(self.keywords[i]) > self.keyword_size: 
            # augmentation
            keyword = [self.keywords[i][0]]
            random_keys = self.keywords[i][1:]
            random.shuffle(random_keys)
            keyword += random_keys[:self.keyword_size-1]
        else:
            keyword = self.keywords[i]

        keyword = torch.LongTensor(keyword)    
        caption = torch.LongTensor(self.captions[i])
        caplen = torch.LongTensor([self.caplens[i]])

        if self.split is 'TRAIN':
            return keyword, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return keyword, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size
