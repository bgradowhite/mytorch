import torch
import torch.nn as nn
from torch import optim
import mytorch.layers as ml
import mytorch.functions as mf
from tqdm import tqdm
import matplotlib.pyplot as plt

def build_mlp( input_size, output_size, n_size, device=None, dtype=None)->nn.Module:
    """
        Builds MLP
        arguments:
            n_size: list of the sizes of each hidden layer
            input_size: size of the input layer
            output_size: size of the output layer
        returns:
            mlp (nn.ModuleList)
    """
    kwargs = {'device': device, 'dtype': dtype}
    layers = []
    in_size = input_size
    for size in n_size:
        layers.append(ml.Linear(in_size, size,**kwargs))
        layers.append(ml.ReLU())
        in_size = size
    layers.append(ml.Linear(n_size[-1], output_size, **kwargs))
    mlp = nn.ModuleList(layers)
    return mlp

class MLP(nn.Module):
    def __init__(self,input_size, output_size, n_size,learning_rate=1e-4,
                 dtype=None, device =None):
        kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        #init vars
        self.input_size = input_size
        self.output_size = output_size
        self.n_size = n_size
        self.learning_rate = learning_rate
        self.mlp = build_mlp(
            input_size=self.input_size,
            output_size=self.output_size,
            n_size=self.n_size, **kwargs
        )

        self.optimizer = optim.Adam(
            self.mlp.parameters(),
            self.learning_rate
        )
        
    def forward(self, input:torch.Tensor)->torch.Tensor:
        out =input
        for layer in self.mlp:
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
    
    def save_model(self, filepath):
        print("Saving Model to", filepath)
        torch.save(self.state_dict(), filepath)

    def train_model(self, data, epochs:int, batch_size:int, device, lr_schedule=False, save=None):
        train_data, val_data, test_data = data
        X,Y = train_data
        n = len(Y)
        train_loss, val_acc = [],[]
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
                info = self.update(X_b, Y_b)
                train_loss.append(info['Training Loss'])
                bar.set_description(f'Training Loss {info['Training Loss']:.2f}')
            val_acc.append(self.get_error(val_data, batch_size))

        if save:
            self.save_model(save)
        test_acc = self.get_error(test_data, batch_size)
        print(test_acc[1])
        
        plt.plot(train_loss, color='blue')
        plt.xlabel('training examples')
        plt.ylabel('loss')
        plt.show()
