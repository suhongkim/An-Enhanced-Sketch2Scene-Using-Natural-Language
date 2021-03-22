import torch
import torch.nn.functional as F
import numpy as np
import json
import argparse
import pymagnitude
import os
import random

def sketch2caption_all(base_name, keyword_size, w2v_magnitdue_path, beam_size, num_sen): 
    # Evaluation on Sketch2Caption 
    with open(os.path.join('data', 'doodle_classes.json'), 'r') as j:
        doodle = json.load(j)
        
    outputs = {}
    failure_class = []
    for dd in doodle: 
        keys, sentences, failure = sketch2caption(doodle_class=dd, 
                    checkpoint='pretrained/BEST_checkpoint_coco_{}_{}.pth.tar'.format(base_name, keyword_size), 
                    word_map_path='data/{}/WORDMAP_coco_{}_{}.json'.format(base_name, base_name, keyword_size), 
                    w2v_magnitdue_path=w2v_magnitdue_path, 
                    beam_size =beam_size, num_sen = num_sen)
        if failure: 
            failure_class.append(dd)
        else:
            outputs[dd] = [{'keyword': k, 'caption':' '.join(c)} for k, c in zip(keys, sentences)]
        
    with open(os.path.join('outputs', 'outputs3_{}_{}.json'.format(base_name, keyword_size)), 'w') as f : 
        json.dump(outputs, f)
        
    print(len(failure_class), failure_class)



def sketch2caption(doodle_class, checkpoint, word_map_path, w2v_magnitdue_path, beam_size=1, num_sen=5):
    # read wordmap 
    with open(word_map_path, 'r') as j: 
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    # read w2v 
    w2v = pymagnitude.Magnitude(w2v_magnitdue_path)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint)
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    
    keyword_size = checkpoint['keyword_size']
    failure = 0
    
    # generate keywords 
    key_candidates = [w[0].lower() for w in w2v.most_similar(doodle_class, topn=keyword_size*10)]
    key_candidates = [w for w in key_candidates if word_map.get(w) is not None]
    # if len(key_candidates) < keyword_size: 
    #     failure = 100

    # Encode, decode with attention and beam search
    sentences = []
    keys = []
    sent_count = 0
    while(sent_count < num_sen):
        random.shuffle(key_candidates)
        key = [doodle_class]
        key += key_candidates[:keyword_size-1]
        if len(key) < keyword_size: 
            key = [doodle_class]*keyword_size 
        seq = caption_beam_search(decoder, key, word_map, beam_size)
        unk_count =[s for s in seq if s in {word_map['<unk>']}]
        if len(seq) == 0: # or len(unk_count) > 0: 
            # print('Caption is not generated on ', key)
            failure += 1 
            if failure > num_sen*10: 
                break
            else:
                continue
        sentences.append([rev_word_map.get(s) for s in seq if s not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
        keys.append(key)
        sent_count += 1

    return keys, sentences, failure>0


def caption_beam_search(decoder, keys, word_map, beam_size):
    """
    Reads an image and captions it with beam search.
    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k = beam_size
    vocab_size = len(word_map)

    # Read keywords               
    encoded_keywords = torch.LongTensor([word_map.get(key, word_map['<unk>']) for key in keys]).to(device)
    key_embedding = decoder.embedding(encoded_keywords)
    encoder_dim = key_embedding.shape[0]*key_embedding.shape[1]

    # Flatten encoding
    encoder_out = key_embedding.view(1, -1, encoder_dim)  # (1, 1, encoder_dim)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, 1, encoder_dim)  # (k, 1, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out.squeeze(1))

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
        encoder_out = encoder_out.view(-1, encoder_dim)  # (1, 1, encoder_dim)

        h, c = decoder.decode_step(torch.cat([embeddings, encoder_out], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds] / seqs[complete_inds].shape[1])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    if len(complete_seqs_scores) == 0: 
        return []
    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]

    if len(seq) < 5: 
        # print("the length of the top seq is less than 5", complete_seqs_scores)
        return []
    return seq

if __name__ == '__main__':
    # # get doodle class name 
    with open(os.path.join('data', 'doodle_classes.json'), 'r') as j:
        doodle = json.load(j)
    doodle_class = random.choice(doodle) 

    keys, sentences, failure = sketch2caption(doodle_class=doodle_class, 
                    checkpoint='pretrained/BEST_checkpoint_coco_retrofit_5.pth.tar', 
                    word_map_path='data/retrofit/WORDMAP_coco_retrofit_5.json', 
                    w2v_magnitdue_path='data/glove.42B.300d.retrofit.magnitude', 
                    beam_size=3, num_sen=5)
    print("Doodle Class Name (Randomly Selected): ", doodle_class)
    for i in range(len(keys)):
        print(keys[i], ' '.join(sentences[i]))

    # Generate All ouputs 
    # sketch2caption_all(base_name='baseline2', 
    #                 keyword_size=3, 
    #                 w2v_magnitdue_path='data/glove.42B.300d.magnitude', 
    #                 beam_size=5, num_sen=5)

