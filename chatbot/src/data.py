from typing import Any
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torchtext.data.functional import to_map_style_dataset
import torchtext
import torch
import nltk
import string

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stemmer = nltk.stem.snowball.SnowballStemmer('english')

class Seq2SeqDataset(Dataset):

    def __init__(self):

        self.data = []
        self.data_path = ""
        self.source_data = None
        self.target_data = None
        self.vocab = None
        self.source_tensors = None
        self.target_tensors = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.source_tensors[item], self.target_tensors[item]

    def tokenizer(self, text):
        # Extract the text from the context and question fields
        sentence = ''.join([s.lower() for s in text if s not in string.punctuation])
        sentence = ' '.join(stemmer.stem(w) for w in sentence.split())
        tokens = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(sentence)
        return tokens
    
    def to_tensor(self, sentence, vocab):
        try:
            indices = [vocab.word2index[word] for word in sentence]
        except KeyError:
            indices = [vocab.word2index["<unk>"] if word not in vocab.word2index.keys() else vocab.word2index[word] for word in sentence]
                      
        indices.append(vocab.word2index["<eos>"])
        return torch.Tensor(indices).long().to(device)

    def get_pairs(self, dataset=None):
        if dataset is None:
            dataset = self.data
        source_data = []
        target_data = []
        for data in dataset:
            source_data.append(data[1])
            target_data.append(data[2][0])
        self.source_data = source_data
        self.target_data = target_data
        return self   

    def load_train_data(self, path=None):
        dev, test = torchtext.datasets.SQuAD1()
        dev = to_map_style_dataset(dev)
        self.data = dev._data[0:10000:2]
        # self.data = dev._data[1:10000]
        return self
    
    def load_val_data(self, path=None):
        dev, test = torchtext.datasets.SQuAD1()
        dev = to_map_style_dataset(dev)
        self.data = dev._data[1:6000:4]
        # self.data = dev._data[10000:12000]
        return self    

    def tokenize(self):
        self.source_tokens = list(map(self.tokenizer, self.source_data))
        self.target_tokens = list(map(self.tokenizer, self.target_data))
        return self

    def numerize(self, vocab):
        source_tensor = []
        for src in self.source_tokens:
            source_tensor.append(self.to_tensor(src, vocab))
        target_tensor = []
        for trg in self.target_tokens:
            target_tensor.append(self.to_tensor(trg, vocab))
        self.source_tensors = source_tensor
        self.target_tensors = target_tensor
        return self


import gensim
from torchtext.vocab import GloVe
class Vocabulary():

    def __init__(self) -> None:
        self.word_count = 0
        self.word2index = {}
        self.index2word = {}
        self.specials = ["<bos>", "<eos>", "<unk>", "<pad>"]
    

    def from_iterator(self, dataset):

        vocab = torchtext.vocab.build_vocab_from_iterator(
            map(dataset.tokenizer, [pair[1]+pair[2][0] for pair in dataset.data]),
            min_freq=2,  # Set the minimum frequency of each word to 2
            specials=["<bos>", "<eos>", "<unk>", "<pad>"],  # Add special tokens
            special_first=True)
        vocab.set_default_index(vocab["<unk>"])
        self.word_count = len(vocab)
        self.word2index = vocab.get_stoi()
        self.index2word = {index: word for word,index in self.word2index.items()}
        return self