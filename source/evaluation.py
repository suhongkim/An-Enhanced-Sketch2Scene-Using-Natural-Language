import os
import sys

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm

import sacrebleu

def selfbleu_sacrebleu(base_name, keyword_size): 
    with open(os.path.join('outputs','outputs3_{}_{}.json'.format(base_name, keyword_size)), 'r') as o: 
        sketch_captions = json.load(o)
        
    total_score = []   
    for key in sketch_captions.keys():
        refs = []
        for c in sketch_captions[key]: 
            refs.append([c['caption']])
        score = []
        for c in sketch_captions[key]: 
            hyp = [c['caption']]
            ref = [r for r in refs if r != hyp]
            bleu = sacrebleu.corpus_bleu(hyp, ref, force=True, lowercase=True, tokenize='none')
            score.append(bleu.score)
        total_score.append(sum(score) / len(score))
    return (sum(total_score) / len(total_score))


def selfbleu(base_name, keyword_size):
    with open(os.path.join('outputs','outputs3_{}_{}.json'.format(base_name, keyword_size)), 'r') as o: 
        sketch_captions = json.load(o)
        
    total_score = []   
    for key in sketch_captions.keys():
        score = []
        captions = [c['caption'] for c in sketch_captions[key]]
        for i in range(len(captions)): 
            hyp = captions[i].split()
            refs = [captions[j].split() for j in range(len(captions)) if j != i]
            bleu = corpus_bleu([refs], [hyp])
            score.append(bleu)
        total_score.append(sum(score) / len(score))
    # print('Self-Bleu', sum(total_score) / len(total_score))
    return sum(total_score) / len(total_score)


def semantic_acc_class(base_name, keyword_size): 
    with open(os.path.join('outputs','outputs3_{}_{}.json'.format(base_name, keyword_size)), 'r') as o: 
        sketch_captions = json.load(o)
    
    total_acc = []   
    for key in sketch_captions.keys():
        count  = 0
        captions = [c['caption'] for c in sketch_captions[key]]
        for c in captions: 
            if key in c.split(): 
                count += 1           
        total_acc.append(count / len(captions))

    print('Semantic accuracy', sum(total_acc) / len(total_acc))
    return sum(total_acc) / len(total_acc)

def semantic_acc_keys(base_name, keyword_size): 
    with open(os.path.join('outputs','outputs3_{}_{}.json'.format(base_name, keyword_size)), 'r') as o: 
        sketch_captions = json.load(o)
        
    total_acc = []   
    for key in sketch_captions.keys():
        count  = 0
        keys = [c['keyword'] for c in sketch_captions[key]]
        captions = [c['caption'] for c in sketch_captions[key]]
        for k, c in zip(keys,captions): 
            intersection = list(set(k) & set(c.split()))
            if len(intersection) > 0: 
                count += 1           
        total_acc.append(count / len(captions))

    # print('Semantic accuracy', sum(total_acc) / len(total_acc))
    return sum(total_acc) / len(total_acc)


def evaluate(base_name, keyword_size, checkpoint, beam_size=1):
    """
    Evaluation
    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """

    data_folder = os.path.join('data',base_name)  # folder with data files saved by create_input_files.py
    data_name = 'coco_' + base_name + '_' + str(keyword_size)  # base name shared by data files
    word_map_file = os.path.join('data',base_name,'WORDMAP_coco_' + base_name + '_' + str(keyword_size)+'.json')  # word map, ensure it's the same the data was encoded with and the model was trained with
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
    cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

    # Load word map (word2ix)
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}
    vocab_size = len(word_map)

    # Load model
    checkpoint = torch.load(checkpoint)
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()

    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, keyword_size, 'TEST'),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
    ref_inwords = list()
    hyp_inwords = list()
    key_inwords = list()
    
    # For each image
    for i, (keys, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size

        # Move to GPU device, if available
        keys = keys.to(device)  # (1, 3, 256, 256)
        
        # Read keywords                     
        key_embedding = decoder.embedding(keys)
        encoder_dim = key_embedding.shape[1]*key_embedding.shape[2]

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

        # Lists to store completed sequences and scores
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

            # Add new words to sequences
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
            continue
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        references.append(img_captions)
        ref_inwords.append(list(
            map(lambda c: [rev_word_map[w] for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps)))

        # Hypotheses
        hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
        hyp_inwords.append([rev_word_map[w] for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
        key_inwords.append([rev_word_map[k] for k in keys[0].tolist()])
        # if len(hypotheses) != len(references): 
        #     print(hypotheses)
        # if len(hyp_inwords[-1])== 0: 
        # print([rev_word_map[k] for k in keys[0].tolist()])
        # print(ref_inwords[-1])
        # print(hyp_inwords[-1])
        # print(len(references[-1]),len(hypotheses[-1]), len(seq))

        assert len(references) == len(hypotheses)

    # Calculate BLEU-4 scores
    # bleu4 = corpus_bleu(references, hypotheses)

    return references, hypotheses, ref_inwords, hyp_inwords, key_inwords


def bleu(ref_t, pred_t):
    return sacrebleu.corpus_bleu(pred_t, ref_t, force=True, lowercase=True, tokenize='none')

if __name__ == '__main__':
    base_name, keyword_size = 'baseline',  3

    print("Evaluation for Basline-1 model with input 3 conditional input words")
    checkpoint = os.path.join('pretrained','BEST_checkpoint_coco_{}_{}.pth.tar'.format(base_name, keyword_size))
    references, hypotheses, ref_inwords, hyp_inwords, key_inwords = evaluate(base_name, keyword_size, checkpoint)
    print("BLEU4 Score: ", corpus_bleu(references, hypotheses))
    print("Self -BLEU4 Score: ",selfbleu(base_name, keyword_size))
    print("Semantic Accuracy Score: ",semantic_acc_keys(base_name, keyword_size))

    base_name, keyword_size = 'baseline',  5

    print("Evaluation for Basline-1 model with input 5 conditional input words")
    checkpoint = os.path.join('pretrained','BEST_checkpoint_coco_{}_{}.pth.tar'.format(base_name, keyword_size))
    references, hypotheses, ref_inwords, hyp_inwords, key_inwords = evaluate(base_name, keyword_size, checkpoint)
    print("BLEU4 Score: ", corpus_bleu(references, hypotheses))
    print("Self -BLEU4 Score: ",selfbleu(base_name, keyword_size))
    print("Semantic Accuracy Score: ",semantic_acc_keys(base_name, keyword_size))

    base_name, keyword_size = 'baseline2',  3

    print("Evaluation for Basline-2 model with input 3 conditional input words")
    checkpoint = os.path.join('pretrained','BEST_checkpoint_coco_{}_{}.pth.tar'.format(base_name, keyword_size))
    references, hypotheses, ref_inwords, hyp_inwords, key_inwords = evaluate(base_name, keyword_size, checkpoint)
    print("BLEU4 Score: ", corpus_bleu(references, hypotheses))
    print("Self -BLEU4 Score: ",selfbleu(base_name, keyword_size))
    print("Semantic Accuracy Score: ",semantic_acc_keys(base_name, keyword_size))

    base_name, keyword_size = 'baseline2',  5

    print("Evaluation for improved model with input 5 conditional input words")
    checkpoint = os.path.join('pretrained','BEST_checkpoint_coco_{}_{}.pth.tar'.format(base_name, keyword_size))
    references, hypotheses, ref_inwords, hyp_inwords, key_inwords = evaluate(base_name, keyword_size, checkpoint)
    print("BLEU4 Score: ", corpus_bleu(references, hypotheses))
    print("Self -BLEU4 Score: ",selfbleu(base_name, keyword_size))
    print("Semantic Accuracy Score: ",semantic_acc_keys(base_name, keyword_size))

    base_name, keyword_size = 'retrofit',  3

    print("Evaluation for improved model with input 3 conditional input words")
    checkpoint = os.path.join('pretrained','BEST_checkpoint_coco_{}_{}.pth.tar'.format(base_name, keyword_size))
    references, hypotheses, ref_inwords, hyp_inwords, key_inwords = evaluate(base_name, keyword_size, checkpoint)
    print("BLEU4 Score: ", corpus_bleu(references, hypotheses))
    print("Self -BLEU4 Score: ",selfbleu(base_name, keyword_size))
    print("Semantic Accuracy Score: ",semantic_acc_keys(base_name, keyword_size))

    base_name, keyword_size = 'retrofit',  5

    print("Evaluation for improved model with input 5 conditional input words")
    checkpoint = os.path.join('pretrained','BEST_checkpoint_coco_{}_{}.pth.tar'.format(base_name, keyword_size))
    references, hypotheses, ref_inwords, hyp_inwords, key_inwords = evaluate(base_name, keyword_size, checkpoint)
    print("BLEU4 Score: ", corpus_bleu(references, hypotheses))
    print("Self -BLEU4 Score: ",selfbleu(base_name, keyword_size))
    print("Semantic Accuracy Score: ",semantic_acc_keys(base_name, keyword_size))

    base_name, keyword_size = 'augment',  3

    print("Evaluation for improved model + data augmentation with input 3 conditional input words")
    checkpoint = os.path.join('pretrained','BEST_checkpoint_coco_{}_{}.pth.tar'.format(base_name, keyword_size))
    references, hypotheses, ref_inwords, hyp_inwords, key_inwords = evaluate(base_name, keyword_size, checkpoint)
    print("BLEU4 Score: ", corpus_bleu(references, hypotheses))
    print("Self -BLEU4 Score: ",selfbleu(base_name, keyword_size))
    print("Semantic Accuracy Score: ",semantic_acc_keys(base_name, keyword_size))

    base_name, keyword_size = 'augment',  5

    print("Evaluation for improved model + data augmentation with input 5 conditional input words")
    checkpoint = os.path.join('pretrained','BEST_checkpoint_coco_{}_{}.pth.tar'.format(base_name, keyword_size))
    references, hypotheses, ref_inwords, hyp_inwords, key_inwords = evaluate(base_name, keyword_size, checkpoint)
    print("BLEU4 Score: ", corpus_bleu(references, hypotheses))
    print("Self -BLEU4 Score: ",selfbleu(base_name, keyword_size))
    print("Semantic Accuracy Score: ",semantic_acc_keys(base_name, keyword_size))




