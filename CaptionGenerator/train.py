# https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

import os
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Decoder
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu


class Trainer: 
    def __init__(self, base_name, keyword_size, epochs=200, batch_size=64, workers=1, decoder_lr=4e-4, checkpoint=None):
        """
        Training and validation.
        """
        self.base_name = base_name 
        self.keyword_size = keyword_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.workers = workers
        self.decoder_lr = decoder_lr
        self.checkpoint = checkpoint
        # path 
        self.data_folder = os.path.join('data', base_name)  # folder with data files saved by create_input_files.py
        self.data_name = 'coco_{}_{}'.format(base_name, str(keyword_size))
        self.out_folder = 'pretrained' # output dir of model 

        # Model parameters
        self.emb_dim = 512  # dimension of word embeddings
        self.decoder_dim = 512  # dimension of decoder RNN
        self.dropout = 0.5
        self.print_freq = 100
        self.grad_clip = 5.  # clip gradients at an absolute value of
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
        cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

        # Read word map
        self.word_map_file = os.path.join(self.data_folder, 'WORDMAP_' + self.data_name + '.json')
        with open(self.word_map_file, 'r') as j:
            self.word_map = json.load(j)

        # Initialize / load checkpoint
        if self.checkpoint is None:
            self.keyword_size = keyword_size
            self.start_epoch = 0  
            self.epochs_since_improvement = 0   
            self.best_bleu4 = 0.  
            self.decoder = Decoder(embed_dim=self.emb_dim, decoder_dim=self.decoder_dim,
                            vocab_size=len(self.word_map), keyword_size=self.keyword_size, 
                            dropout=self.dropout)
            self.decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, self.decoder.parameters()),
                                                lr=self.decoder_lr)

        else:
            self.keyword_size = self.check['keyword_size']
            print('keyowrd_size: ', keyword_size)
            self.checkpoint = torch.load(self.checkpoint)
            self.start_epoch = self.checkpoint['epoch'] + 1
            self.epochs_since_improvement = self.checkpoint['epochs_since_improvement']
            self.best_bleu4 = self.checkpoint['bleu-4']
            self.decoder = self.checkpoint['decoder']
            self.decoder_optimizer = self.checkpoint['decoder_optimizer']

        # Move to GPU, if available
        self.decoder = self.decoder.to(self.device)

        # Loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        # Custom dataloaders
        self.train_loader = torch.utils.data.DataLoader(CaptionDataset(self.data_folder,self.data_name, self.keyword_size, 'TRAIN'),
                                                    batch_size=self.batch_size, shuffle=True, num_workers=self.workers, pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(CaptionDataset(self.data_folder, self.data_name, self.keyword_size, 'VAL'),
                                                batch_size=self.batch_size, shuffle=True, num_workers=self.workers, pin_memory=True)

    def run(self):
        # Epochs
        for epoch in range(self.start_epoch, self.epochs):

            # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
            if self.epochs_since_improvement == 5:
                break
            if self.epochs_since_improvement > 0 and self.epochs_since_improvement % 3 == 0:
                adjust_learning_rate(self.decoder_optimizer, 0.8)

            # One epoch's training
            self.train(epoch)

            # One epoch's validation
            recent_bleu4 = self.validate()

            # Check if there was an improvement
            is_best = recent_bleu4 > self.best_bleu4
            self.best_bleu4 = max(recent_bleu4, self.best_bleu4)
            if not is_best:
                self.epochs_since_improvement += 1
                print("\nEpochs since last improvement: %d\n" % (self.epochs_since_improvement,))
            else:
                self.epochs_since_improvement = 0

            # Save checkpoint
            save_checkpoint(self.out_folder, self.data_name, epoch, self.epochs_since_improvement, self.decoder, 
                            self.decoder_optimizer, self.keyword_size, recent_bleu4, is_best)


    def train(self, epoch):
        self.decoder.train()  # train mode (dropout and batchnorm is used)

        batch_time = AverageMeter()  # forward prop. + back prop. time
        data_time = AverageMeter()  # data loading time
        losses = AverageMeter()  # loss (per word decoded)
        top5accs = AverageMeter()  # top5 accuracy

        start = time.time()

        # Batches
        for i, (keys, caps, caplens) in enumerate(self.train_loader):
            data_time.update(time.time() - start)

            # Move to GPU, if available
            keys = keys.to(self.device)
            caps = caps.to(self.device)
            caplens = caplens.to(self.device)

            # Forward prop.
            scores, caps_sorted, decode_lengths, sort_ind = self.decoder(keys, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            packedScores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            scores = packedScores.data
            packedTargets = pack_padded_sequence(targets, decode_lengths, batch_first=True)
            targets = packedTargets.data

            # Calculate loss
            loss = self.criterion(scores, targets)

            # Back prop.
            self.decoder_optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            if self.grad_clip is not None:
                clip_gradient(self.decoder_optimizer, self.grad_clip)

            # Update weights
            self.decoder_optimizer.step()

            # Keep track of metrics
            top5 = accuracy(scores, targets, 5)
            losses.update(loss.item(), sum(decode_lengths))
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i % self.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(self.train_loader),
                                                                            batch_time=batch_time,
                                                                            data_time=data_time, loss=losses,
                                                                            top5=top5accs))

    def validate(self):
        self.decoder.eval()  # eval mode (no dropout or batchnorm)

        batch_time = AverageMeter()
        losses = AverageMeter()
        top5accs = AverageMeter()

        start = time.time()

        references = list()  # references (true captions) for calculating BLEU-4 score
        hypotheses = list()  # hypotheses (predictions)

        # explicitly disable gradient calculation to avoid CUDA memory error
        # solves the issue #57
        with torch.no_grad():
            # Batches
            for i, (keys, caps, caplens, allcaps) in enumerate(self.val_loader):

                # Move to self.device, if available
                keys = keys.to(self.device)
                caps = caps.to(self.device)
                caplens = caplens.to(self.device)

                # Forward prop.
                scores, caps_sorted, decode_lengths, sort_ind = self.decoder(keys, caps, caplens)

                # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
                targets = caps_sorted[:, 1:]

                # Remove timesteps that we didn't decode at, or are pads
                # pack_padded_sequence is an easy trick to do this
                scores_copy = scores.clone()
                packedScores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
                scores = packedScores.data
                packedTargets = pack_padded_sequence(targets, decode_lengths, batch_first=True)
                targets = packedTargets.data

                # Calculate loss
                loss = self.criterion(scores, targets)

                # Keep track of metrics
                losses.update(loss.item(), sum(decode_lengths))
                top5 = accuracy(scores, targets, 5)
                top5accs.update(top5, sum(decode_lengths))
                batch_time.update(time.time() - start)

                start = time.time()

                if i % self.print_freq == 0:
                    print('Validation: [{0}/{1}]\t'
                        'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(self.val_loader), batch_time=batch_time,
                                                                                    loss=losses, top5=top5accs))

                # References
                allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
                for j in range(allcaps.shape[0]):
                    img_caps = allcaps[j].tolist()
                    img_captions = list(
                        map(lambda c: [w for w in c if w not in {self.word_map['<start>'], self.word_map['<pad>']}],
                            img_caps))  # remove <start> and pads
                    references.append(img_captions)

                # Hypotheses
                _, preds = torch.max(scores_copy, dim=2)
                preds = preds.tolist()
                temp_preds = list()
                for j, p in enumerate(preds):
                    temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
                preds = temp_preds
                hypotheses.extend(preds)

                assert len(references) == len(hypotheses)

            # Calculate BLEU-4 scores
            bleu4 = corpus_bleu(references, hypotheses)

            print(
                '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                    loss=losses,
                    top5=top5accs,
                    bleu=bleu4))

        return bleu4


if __name__ == '__main__':

    trainer = Trainer(base_name='retrofit', keyword_size=5, 
                    epochs=10, batch_size=64, workers=1, decoder_lr=4e-4, 
                    checkpoint=None )
    trainer.run()


