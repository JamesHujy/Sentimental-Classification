import codecs
import matplotlib.pyplot as plt 
import pickle
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
import tqdm
import torch.nn.functional as F


class dataset(Dataset):
    def __init__(self, args, train=True):
        self.label = []
        self.Label = []
        self.weight = []
        self.text = []
        self.token = []
        self.idx2word = []
        self.length = []
        self.embedweights = []
        self.word2idx = {}
        self.vector = {}
        if not train:
            self.buildtestvocab(classify=args.classify)
        else:
            self.build_vocab(classify=args.classify)
            self.build_embed()
            self.build_text()

    def __getitem__(self, item):
        data = torch.Tensor(self.token[item])
        length = np.array(self.length)
        return data, self.Label[item], length[item]

    def __len__(self):
        return len(self.label)

    def labelweight(self):
        return None

    def read_data(self, file_name='sinanews.train', classify=True):
        print('reading the data')
        with codecs.open('./data/'+file_name,'r',encoding='utf-8') as f:
            for lines in f.readlines():
                line = lines.split()
                templabel = line[2:10]
                text = line[11:]
                label = [int(i.split(':')[-1]) for i in templabel]
                self.label.append(label)
                self.text.append(text)
        if classify:
            temp = []
            for label in self.label:
                maxlabel = 0
                max_index = 0
                for index in range(len(label)):
                    if label[index] > maxlabel:
                        maxlabel = label[index]
                        max_index = index
                temp.append(max_index)
            self.Label = np.array(temp)

        else:
            labellist = []
            for index in range(len(self.label)):
                label = self.label[index]
                sum = 0
                for sub in label:
                    sum += int(sub)
                temp = []
                for sub in label:
                    try:
                        temp.append(sub/sum)
                    except:
                        temp.append(0)
                labellist.append(temp)
            self.Label = np.array(labellist)


    def calculength(self):
        length = [len(text) for text in self.text]
        plt.hist(length, bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
        plt.title("hist of the sentences' length")
        plt.show()

    def read_vector(self,file_name='sgns.sogounews.bigram-char'):
        print('load_vector...')
        vector = {}
        try:
            with open('./data/wordvector.pkl','rb') as f:
                vector = pickle.load(f)
                return vector
        except:
            print("loading...")
            with codecs.open('./data/'+file_name,'r',encoding='utf-8') as f:
                for lines in f.readlines():
                    line = lines.split()
                    vector[line[0]] = line[1:]
            with open('./data/wordvector.pkl','wb') as f:
                pickle.dump(self.vector,f)
        return vector

    def build_vocab(self, file_name='sinanews.train',classify=True):
        print('build_vocab')
        self.read_data(file_name=file_name,classify=classify)
        try:
            with open('./data/idx2word.pkl', 'rb') as f:
                self.idx2word = pickle.load(f)

            with open('./data/word2idx.pkl', 'rb') as f:
                self.word2idx = pickle.load(f)

            print(len(self.idx2word))
        except:
            #self.vector = self.read_vector()
            self.idx2word.append('<unk>')
            for text in self.text:
                for word in text:
                    self.idx2word.append(word)
            for word in tqdm.tqdm(self.vector.keys()):
                self.idx2word.append(word)
            self.idx2word = list(set(self.idx2word))
            index = 0
            for word in self.idx2word:
                self.word2idx[word] = index
                index += 1
            with open('./data/idx2word.pkl', 'wb') as f:
                pickle.dump(self.idx2word, f)
            with open('./data/word2idx.pkl', 'wb') as f:
                pickle.dump(self.word2idx, f)

    def build_text(self, max_length=1500):
        print('build_text')
        for text in self.text:
            temp = []
            for word in text:
                try:
                    temp.append(self.word2idx[word])
                except:
                    temp.append(self.word2idx['<unk>'])

            length = len(temp)
            if length > max_length:
                self.length.append(max_length)
                length = max_length
                temp = temp[:max_length]
            else:
                self.length.append(length)
            length = max_length - length
            for i in range(length):
                temp.append(0)
            self.token.append(temp)
        self.token = np.array(self.token)

    def build_embed(self):
        print('build_embed')
        try:
            with open('./data/embedmatrix.pkl', 'rb') as f:
                self.embedweights = pickle.load(f)
        except:
            self.embedweights.append(np.zeros(300))
            for word in tqdm.tqdm(self.idx2word):
                try:
                    self.embedweights.append(self.vector[word])
                except:
                    temp = []
                    for i in range(300):
                        rand = random.uniform(-0.5, 0.5)
                        temp.append(rand)
                    self.embedweights.append(temp)
            with open('./data/embedmatrix.pkl', 'wb') as f:
                pickle.dump(self.embedweights, f)

    def getembed(self):
        return self.embedweights

    def buildtestvocab(self, classify=True):
        self.build_vocab(file_name='sinanews.test',classify=classify)
        self.build_text()

