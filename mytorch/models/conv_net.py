import torch, math
from tqdm.auto import tqdm
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import mytorch.functions as mf
import mytorch.layers as ml



def build_convnet(n_channels,n_fc, dropout = None,device=None, dtype=None)->nn.Module:
    """
        Builds a convolutional network 
        arguments:
            n_channels: list of the number of filters 
            n_classes:
            n_fc: sizes of fully connected layers after convolutions
            dropout: None or probabilty for dropout layer
        returns:
            convnet (nn.ModuleList)
    """
    kwargs = {'device': device, 'dtype': dtype}
    layers = []
    in_size = n_channels[0]
    for i in range(len(n_channels)-1):
        layers.append(ml.Conv2d(n_channels[i],n_channels[i+1],kernel_size = (3,3), padding=1, **kwargs))
        layers.append(ml.ReLU())
        if i%2==1:
            layers.append(ml.MaxPool(kernel_size = (2,2),stride=2))
    layers.append(ml.Flatten(start=1))
    n_in = n_channels[-1]*8*8
    for i,n_out in enumerate(n_fc):
        layers.append(ml.Linear(n_in,n_out,**kwargs))
        n_in = n_out
        if i!=len(n_fc)-1:
            layers.append(ml.ReLU())
            layers.append(ml.Dropout(dropout))
        
    convnet = nn.ModuleList(layers)
    return convnet

class ConvNet(nn.Module):
    def __init__(self, n_channels,n_fc, dropout=None,learning_rate = 1e-3,device=None, dtype=None):
        super().__init__()
        kwargs = {"device": device, "dtype": dtype}
        #init vars
        self.n_channels = n_channels
        self.n_classes = n_fc[-1]
        self.learning_rate = learning_rate
        self.dropout = dropout

        self.convnet = build_convnet(
            self.n_channels,
            n_fc,
            dropout,
            **kwargs
        )

        self.optimizer = optim.Adam(
            self.convnet.parameters(),
            self.learning_rate
        )
     

    def forward(self, input):
        """
        input of size (batch, in_channels,in_height, in_width )
        output of size (batch, classes)
        """
        out =input
        for layer in self.convnet:
            out = layer(out) 
        return out
  

    @torch.no_grad()
    def get_error(self, data, batch_size):
        self.eval()
        X,Y = data
        test_loss, cor = 0,0
        size = len(Y)
        n_b = size//batch_size

        for i in range(n_b):
            X_b = X[i*batch_size:i*batch_size+ batch_size]
            Y_b = Y[i*batch_size:i*batch_size + batch_size]
            pred = self(X_b)
            test_loss += mf.cross_entropy(pred,Y_b).item()
            cor+=(pred.argmax(1)==Y_b).type(torch.float).sum().item()
        test_loss/=n_b
        cor/=size
        print(f"Test Error: \n Accuracy: {(100*cor):>0.1f}%, Avg loss: {test_loss:>8f}\n")
        return test_loss, cor


    def update(self,input, label):
        logits = self(input)
        self.optimizer.zero_grad()
        loss = mf.cross_entropy(logits, label)
        loss.backward()
        self.optimizer.step()
        return {
            'Training Loss': loss.item(),
        }

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def train_model(self, data,epochs, batch_size, save=None):
        train_data, val_data, test_data = data
        X,Y = train_data
        n = len(Y)
        steps, train_loss, val_acc = [],[],[]
        step = 0

        n_b = n//batch_size
        bar = tqdm(range(epochs),total=epochs)
        for _ in bar:
            self.train()
            #shuffle data and batch
            perm = torch.randperm(n) 
            X,Y= X[perm], Y[perm]
            for i in range(n_b):
                X_b = X[i*batch_size:i*batch_size + batch_size]
                Y_b = Y[i*batch_size:i*batch_size+ batch_size]
                if X_b.shape[0]!=batch_size:
                    print("here", X_b.shape, i, n,n_b)
                info = self.update(X_b, Y_b)
                train_loss.append(info['Training Loss'])
                bar.set_description(f'Training Loss {info['Training Loss']:.2f}')
            val_acc.append(self.get_error(val_data, batch_size))

        if save:
            self.save(save)
        print("Final Test Error")
        test_acc = self.get_error(test_data, batch_size)
        print(test_acc[1])
        
        fig = plt.figure()
        plt.plot(train_loss, color='blue')
        plt.xlabel('training examples')
        plt.ylabel('loss')
        plt.show()
