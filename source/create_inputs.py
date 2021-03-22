import os
import numpy as np
import h5py
import json
import torch
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
import nltk
import pymagnitude

def create_doodle_vocab(doodle_class_path, w2v_magnitdue_path, wordmap_path, out_doodle_path, topn=10):
    # Read W2v retrofitted 
    wv = pymagnitude.Magnitude(w2v_magnitdue_path)
    
    # read doodle 
    with open(doodle_class_path, 'r') as j: 
        doodle = json.load(j)
    
    # Build doodle+ list X 10
    doodle_plus = []
    for d in doodle:   
        doodle_plus.append(d)
        doodle_plus += set([k[0].lower() for k in wv.most_similar(d, topn=topn)])

    # Read wordmap 
    with open(wordmap_path, 'r') as j: 
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    # find the intersection with vocab 
    doodle_map = {d:word_map.get(d) for d in doodle_plus if word_map.get(d) is not None}

    with open(out_doodle_path, 'w') as j:
        json.dump(doodle_map, j)


def create_input_embeddings(base_name, keyword_size, caption_json_path, doodle_json_path, w2v_magnitdue_path, 
                              min_word_freq=5, max_len=50):
    dataset_name = 'coco_' + base_name
    output_folder = os.path.join('data', base_name)
    captions_per_image = 5 

    # Read Karpathy JSON
    with open(caption_json_path, 'r') as j:
        data = json.load(j)

    # Read doodle JSON
    with open(doodle_json_path, 'r') as j:
        doodle = json.load(j)

    # Read w2v 
    w2v = pymagnitude.Magnitude(w2v_magnitdue_path)

    # Read image paths and captions for each image
    train_keywords = []
    train_image_captions = []
    val_keywords = []
    val_image_captions = []
    test_keywords= []
    test_image_captions = []
    word_freq = Counter()
    exclude_count = 0
    total_count = 0
    for img in data['images']:
        captions = []
        key_freq = Counter()
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])
                keys = [t for t in c['tokens'] if doodle.get(t) is not None]
                if len(keys) > 0: 
                    key_freq.update(keys)

        if len(captions) == 0 or len(key_freq) == 0:
            print("----------> no key match: ", captions[-1])
            continue    

        keywords = [nn for nn, c in key_freq.most_common(keyword_size)]
        total_count += 1
        if len(keywords) < keyword_size: 
            # print(keywords)
            exclude_count += 1
            continue

        if img['split'] in {'train', 'restval'}:
            train_keywords.append(keywords)
            train_image_captions.append(captions)
            
        elif img['split'] in {'val'}:
            val_keywords.append(keywords)
            val_image_captions.append(captions)
            
        elif img['split'] in {'test'}:
            test_keywords.append(keywords)
            test_image_captions.append(captions)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset_name + '_' + str(keyword_size)# + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    print("{} of {} will be excluded".format(exclude_count, total_count))

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for keys, imcaps, split in [(train_keywords, train_image_captions, 'TRAIN'), 
                                (val_keywords, val_image_captions, 'VAL'), 
                                (test_keywords, test_image_captions, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image
            print("\nReading %s images and captions, storing to file...\n" % split)
            
            enc_keywords = []
            enc_captions = []
            caplens = []
            for i in range(len(imcaps)):

                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image

                for j, c in enumerate(captions):
                    if keys[i]  in c: 
                        print("found", keys[i], c)

                    # Encode keywords 
                    enc_k = [word_map.get(key, word_map['<unk>']) for key in keys[i]]

                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_keywords.append(enc_k)
                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Sanity check
            assert len(enc_keywords) == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_KEYWORDS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_keywords, j)

            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)


def create_input_files(base_name, keyword_size, caption_json_path, min_word_freq=5,
                       max_len=50):

    dataset_name = 'coco_' + base_name
    output_folder = os.path.join('data', base_name)
    captions_per_image = 5 

    # Read Karpathy JSON
    with open(caption_json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_keywords = []
    train_image_captions = []
    val_keywords = []
    val_image_captions = []
    test_keywords= []
    test_image_captions = []
    word_freq = Counter()

    for img in data['images']:
        captions = []
        nn_freq = Counter()
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])
            nn_words = [t[0] for t in nltk.pos_tag(c['tokens']) if t[1] == 'NN']
            nn_freq.update(nn_words)

        if len(captions) == 0 or len(nn_freq) < keyword_size:
            continue

        keywords = [nn for nn, c in nn_freq.most_common(keyword_size)]
        
        # print(keywords)
        if img['split'] in {'train', 'restval'}:
            train_keywords.append(keywords)
            train_image_captions.append(captions)
            
        elif img['split'] in {'val'}:
            val_keywords.append(keywords)
            val_image_captions.append(captions)
            
        elif img['split'] in {'test'}:
            test_keywords.append(keywords)
            test_image_captions.append(captions)
    
    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset_name + '_' + str(keyword_size) #+ str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for keys, imcaps, split in [(train_keywords, train_image_captions, 'TRAIN'), 
                                (val_keywords, val_image_captions, 'VAL'), 
                                (test_keywords, test_image_captions, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image
            print("\nReading %s images and captions, storing to file...\n" % split)
            
            enc_keywords = []
            enc_captions = []
            caplens = []

            for i in range(len(imcaps)):
                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image

                for j, c in enumerate(captions):
                    
                    # Encode keywords 
                    enc_k = [word_map.get(key, word_map['<unk>']) for key in keys[i]]

                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_keywords.append(enc_k)
                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Sanity check
            assert len(enc_keywords) == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_KEYWORDS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_keywords, j)

            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)
            

if __name__ == '__main__':
    # nltk.download("maxent_treebank_pos_tagger", download_dir='./venv/nltk_data')
    # nltk.download("maxent_ne_chunker",  download_dir='./venv/nltk_data')
    # nltk.download("punkt",  download_dir='./venv/nltk_data')
    # nltk.download(["tagsets", "universal_tagset"],  download_dir='./venv/nltk_data')
    # nltk.download('averaged_perceptron_tagger', download_dir='./venv/nltk_data')

    # # <baseline1>
    # # Create input files (along with word map)
    # create_input_files(base_name='baseline', keyword_size=3,
    #                    caption_json_path='data/dataset_coco.json')

    # # <baseline2> 
    # create_doodle_vocab(doodle_class_path='data/doodle_classes.json',
    #                     w2v_magnitdue_path='data/glove.840B.300d.magnitude', 
    #                     wordmap_path='data/baseline/WORDMAP_coco_baseline_3.json', 
    #                     out_doodle_path='data/doodle_map.json',
    #                     topn=10)

    # create_input_embeddings(base_name='baseline2', keyword_size = 3, 
    #                    caption_json_path='data/dataset_coco.json',
    #                    doodle_json_path='data/doodle_map.json',
    #                    w2v_magnitdue_path ='data/glove.42B.300d.magnitude')

    # <retrofit> 
    create_doodle_vocab(doodle_class_path='data/doodle_classes.json',
                        w2v_magnitdue_path='data/glove.840B.300d.retrofit.magnitude', 
                        wordmap_path='data/baseline/WORDMAP_coco_baseline_3.json', 
                        out_doodle_path='data/doodle_map_retro.json',
                        topn=10)

    # # <Augmentation> 
    # create_input_embeddings(base_name='augment', keyword_size = 7, 
    #                    caption_json_path='data/dataset_coco.json',
    #                    doodle_json_path='data/doodle_map_retro.json',
    #                    w2v_magnitdue_path ='data/glove.42B.300d.retrofit.magnitude')