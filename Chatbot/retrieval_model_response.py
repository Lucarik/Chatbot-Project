import numpy as np
import torch.nn as nn
import torch
import torch.autograd as autograd
from torch.nn import init
import torch.nn.utils.rnn 
from torch.nn.utils.rnn import pad_sequence
import datetime
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import unicodedata
import string
import re
import random
import os
import itertools
import warnings
warnings.filterwarnings('ignore')

def shuffle_list(list):
    random.shuffle(list)

def load_id(sentence, word_to_id):
    sentence_ids = []

    max_sentence_len = 160
    
    sentence_words = sentence.split()
    if len(sentence_words) > max_sentence_len:
        sentence_words = sentence_words[:max_sentence_len]
    for word in sentence_words:
        if word in word_to_id:
            sentence_ids.append(word_to_id[word])
        else: 
            sentence_ids.append(0) #UNK

    return sentence_ids

class Voc:
    def __init__(self):
        self.vocab = {}
        self.sentences = []
        self.word2id = {}
        self.id2vec = None
        
    def save(self):
        torch.save({
                'voc_dict': self.__dict__,
            }, os.path.join('saveDir', 'save_voc2.tar'))
    
    def load(self, filename):
        checkpoint = torch.load(filename)
        self.__dict__ = checkpoint['voc_dict']
        
class LEncoder(nn.Module):
    def __init__(self, embed_size, hidden_size, layers, bi_dir, vocab_size, id_to_vec, dropout):
        super(LEncoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = layers
        self.direction = bi_dir
        self.vocab_size = vocab_size
        self.id_to_vec = id_to_vec
        self.dropout = dropout
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, dropout=self.dropout,
                            num_layers=self.num_layers)
        
        embedding_weights = torch.FloatTensor(self.vocab_size, self.hidden_size)
        embedding_weights = embedding_weights.to(device)
        for id, vec in self.id_to_vec.items():
            embedding_weights[id] = vec

        self.embedding.weight = nn.Parameter(embedding_weights, requires_grad = False)

    def initHiddenCell(self):
        rand_hidden = autograd.Variable(torch.randn(self.direction * self.num_layers, 1, self.hidden_size))
        rand_cell = autograd.Variable(torch.randn(self.direction * self.num_layers, 1, self.hidden_size))
        return rand_hidden, rand_cell

    def forward(self, input_id, hidden, cell):
        tensor_input = torch.LongTensor([input_id])
        tensor_input = tensor_input.to(device)
        embed_input = self.embedding(tensor_input).view(1, 1, -1)
        output, (hidden, cell) = self.lstm(embed_input, (hidden, cell))
        return output, hidden, cell


class Dual_Encoder(nn.Module):
    def __init__(self, encoder):
        super(Dual_Encoder, self).__init__()

        self.encoder = encoder

        self.input_dim = 5 * self.encoder.direction * self.encoder.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, int(self.input_dim/2)),
            nn.Linear(int(self.input_dim/2), 2)
        )

    def forward(self, s1, s2):

        # init hidden, cell
        h1, c1 = self.encoder.initHiddenCell()
        h2, c2 = self.encoder.initHiddenCell()
        h1 = h1.to(device)
        c1 = c1.to(device)
        h2 = h2.to(device)
        c2 = c2.to(device)

        # input one by one

        for i in range(len(s1)):

            v1, h1, c1 = self.encoder(s1[i], h1, c1)

        for j in range(len(s2)):
            v2, h2, c2 = self.encoder(s2[j], h2, c2)

        # utilize these two encoded vectors
        features = torch.cat((v1,torch.abs(v1 - v2),v2,v1*v2, (v1+v2)/2), 2)
        # features = v1-v2
        output = self.classifier(features)

        return output
    
def creating_model(embed_size, hidden_size, layers, bi_dir, vocab_size, id_to_vec, dropout):

    encoder = LEncoder(
            embed_size = embed_size,
            hidden_size = hidden_size,
            layers = layers,
            bi_dir = bi_dir,
            vocab_size = vocab_size,
            id_to_vec = id_to_vec,
            dropout = dropout)

    dual_encoder = Dual_Encoder(encoder)
    
    return encoder, dual_encoder

def get_response_sample(context, num_responses=1):
    # Set vocab
    fileName = 'saveDir/save_voc2.tar'
    voc = Voc()
    voc.load(fileName)
    word_to_id = voc.word2id
    context_ids = load_id(context, word_to_id)
    i = 0
    best_pos = 0
    best_score = 0
    result = []
    responses = voc.sentences
    best_result = 0
    highest_response = ''
    
    # Set up encoder
    embed_size = 50
    hidden_size = 50
    layers = 1
    bi_dir = 1
    vocab_size = len(voc.vocab)
    id_to_vec = voc.id2vec
    encoder, dual_encoder = creating_model(embed_size, hidden_size, layers, bi_dir, vocab_size, id_to_vec, dropout = 0.1)

    encoder = encoder.to(device)
    dual_encoder = dual_encoder.to(device)
    checkpoint = torch.load('saveDir/retrieval_model.tar')
    encoder_sd = checkpoint['en']
    dual_encoder.load_state_dict(encoder_sd)
    dual_encoder.eval();
    
    if num_responses != 1:
        responses = random.sample(responses, num_responses)
        
    for response in responses:
        if not isinstance(response[0], str) or not response[0]:
            continue
        
        response_ids = load_id(response[0], word_to_id)
        #print(response)
        # Run a training iteration with batch
        output = dual_encoder(context_ids, response_ids)
        output = output.squeeze(0)

        # feed output into softmax to get prob prediction
        sm = nn.Softmax(dim=1)
        res = sm(output.data)[:,1]
        #result.append([response, res.data.tolist()])
        if res > best_result:
            best_result = res
            highest_response = response
        
    return highest_response