import torch, os, pickle,gzip, importlib
import mytorch
import numpy as np
import regex as re


def get_poems(filename):
    mytorch_pth = os.path.dirname(mytorch.__file__)
    file_pth = mytorch_pth + '/data/' + filename
    with open(file_pth, 'r', encoding='utf-8') as file: 
         poems = file.read()
    return poems


def unpickle(file_path):
        with open(file_path, 'rb') as file:
            loaded_data= pickle.load(file, encoding='latin1')
        return loaded_data
def gz_unpickle(file_path):
        with gzip.open(file_path, 'rb') as file:
            loaded_data= pickle.load(file, encoding='latin1')
        return loaded_data

def get_mnist(device):
    """Get MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    Training data: tuple (50k images, 28*28=74 len) and (50k,1) labels
    val and test are image/label tuples with 10k entries
    """

    mytorch_pth = os.path.dirname(mytorch.__file__)
    data_pth = mytorch_pth + '/data/MNIST/mnist.pkl.gz'
    data  = gz_unpickle(data_pth)
    data = [(torch.from_numpy(X).to(device),torch.from_numpy(Y).to(device)) for (X,Y) in data]
    return tuple(data)


def get_CIFAR10(device):
    """Get CIFAR10 data as a tuple containing the training data,
    the validation data, and the test data.

    Training data: tuple (40k images, 3,32,32) and (40k,1) labels
    val and test are image/label tuples with 10k entries
    """
    mytorch_pth = os.path.dirname(mytorch.__file__)
    data_dir = mytorch_pth + '/data/CIFAR10'
    data_file_suffix = "/data_batch_"
    #data
    train_batches = [unpickle(data_dir + data_file_suffix +f"{i}") for i in range(1,5)]
    val_batch = unpickle(data_dir + data_file_suffix +"5")
    test_batch = unpickle(data_dir +"/test_batch")
    torch_train_b=[torch.reshape(torch.from_numpy(batch["data"]),(-1,3,32,32)) for batch in train_batches]
    X_train = (1/255.0)*torch.cat(torch_train_b, dim=0).to(device)
    X_val  = (1/255.0)*torch.reshape(torch.from_numpy(val_batch["data"]),(-1,3,32,32)).to(device)
    X_test  = (1/255.0)*torch.reshape(torch.from_numpy(test_batch["data"]),(-1,3,32,32)).to(device)
    
    #labels and dict from meta
    Y_train = torch.cat([torch.tensor(batch["labels"])for batch in train_batches], dim=0).to(device)
    Y_val = torch.tensor(val_batch["labels"]).to(device)
    Y_test = torch.tensor(test_batch["labels"]).to(device)

    meta_data = unpickle(data_dir + "/batches.meta")
    label_dict = {}
    for i, label in enumerate(meta_data["label_names"]):
        label_dict[i] = label

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), label_dict


class WordTokenizer:
    def __init__(self, input_txt, T, lower=True, device=None, dtype=None):
        self.lower=lower
        self.sequence_length = T
        self.unique_words =set('<unk>')
        if self.lower:
            input_txt=input_txt.lower()
        self.tokens = re.split(r'\b', input_txt)
        self.unique_words.update(self.tokens)
        self.vocab_list = sorted(self.unique_words)
        self.vocab_len = len(self.vocab_list)
        self.vocab = {token:idx for idx, token in enumerate(self.vocab_list)}
        self.inverse_vocab = {idx:token for idx, token in enumerate(self.vocab_list)}
        self.text_ids = torch.tensor([self.vocab[token] for token in self.tokens], device=device, dtype=dtype)
    
    def encode(self,text, unk = "<unk>"):
        if self.lower:
            text = text.lower()
        tokens = re.split(r'\b', text)
        token_ids = [self.vocab.get(token, unk) for token in tokens]
        return token_ids
    
    def decode(self, encoded_text:list):
        temp = []
        for ind in encoded_text:
            temp += [self.inverse_vocab[ind]]
        return ''.join(temp)

    def get_batches(self, int_text, T,b):
        nb = (len(int_text) - T)//b
        ix = torch.randperm(len(int_text) - T)
        x = torch.stack([torch.stack([int_text[i:i+T]
                                    for i in ix[j*b:(j+1)*b]]) for j in range(nb)])
        y = torch.stack([torch.stack([int_text[i+1:i+1+T]
                                    for i in  ix[j*b:(j+1)*b]]) for j in range(nb)])
        return x,y


         

        
        
         
        





