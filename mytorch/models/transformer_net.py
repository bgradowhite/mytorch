import torch, math
import mytorch.functions as mf
import mytorch.layers as ml
import matplotlib.pyplot as plt
from tqdm.auto import tqdm 



class TransNet(torch.nn.Module):
    """
    Decoder only transformer used to make a small langauge model 
    """
    def __init__(self, vocab_size, embed_dim, nheads ,nlayers,nfc, dropout, device=None, dtype=None):
        super().__init__()
        kwargs = {'device': device, 'dtype': dtype}
        self.embed = ml.Embedding(vocab_size, embed_dim, **kwargs)
        self.pe= ml.PositionalEncoder(embed_dim, **kwargs)
        self.trans_block = torch.nn.ModuleList(
                                   ml.TransformerBlock(embed_dim, nheads, nfc, dropout, **kwargs)
                                   for _ in range(nlayers))
        self.do = ml.Dropout(dropout)
        
    def forward(self, input):
        out = self.embed(input)#token embedding
        out = self.do(self.pe(out)) #add positional embedding
        for block in self.trans_block:
            out = block(out)
        out = out@self.embed.embedding.T
        return out

    def update(self,optimizer, input, label):
        optimizer.zero_grad()
        logits = self(input)
        logits = logits.view(-1,logits.shape[-1])
        label= label.view(-1)
        loss = mf.cross_entropy(logits, label)
        loss.backward()
        optimizer.step()
        return {
            'Training Loss': loss.item(),
        }
    
    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)
    


    def train_model(self, tokenizer, T, batch_size, epochs, optimizer, save=None):
        num_batches = (len(tokenizer.text_ids) - T)//batch_size
        train_tokens, train_loss = [],[]
        for _ in range(1, epochs + 1):
            X_bs, Y_bs = tokenizer.get_batches(tokenizer.text_ids, T,batch_size)
            prog_bar = tqdm(range(num_batches), total=num_batches)
            for i in prog_bar:
                x_b, y_b = X_bs[i], Y_bs[i]
                info= self.update(optimizer, x_b,y_b)
                if not train_tokens:
                    train_tokens.append(x_b.shape[0]*x_b.shape[1])
                else:
                    train_tokens.append(train_tokens[-1]+x_b.shape[0]*x_b.shape[1])
                train_loss.append(info['Training Loss'])
                prog_bar.set_description(f'Training Loss {info['Training Loss']:.2f}')
                

        # Plot training loss
        plt.semilogy(train_tokens, train_loss, label='Training Loss')
        plt.xlabel('Training Tokens')
        plt.title(f'Final Perplexity: {round(math.exp(train_loss[-1]),2)}')
        plt.legend()
        plt.show()

        # Save model
        if save:
            self.save_model(save)
    

    @torch.no_grad()
    def generate(self, prompt, ngenerate, T, device=None):
        self.eval()
        inp = prompt.to(device).view(1,-1)
        for _ in range(ngenerate):
            context = inp if inp.shape[0] <= T else inp[-T:]
            output = self(context.view(1,-1))
            probs = mf.softmax(output[:,-1,:], dim=-1)
            sample = torch.multinomial(probs, 1)
            inp = torch.cat((inp,sample), dim=-1)
        return inp.view(-1).tolist()