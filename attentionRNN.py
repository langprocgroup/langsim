import torch
from torch import nn
if torch.cuda.is_available(): import torch.cuda as t
else: import torch as t
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import math, random

class EncoderRNN(nn.Module):
    def __init__(self,  input_size, embed_size, hidden_size, 
                        n_layers=1, dropout=0.5, bidirectional = True):
        
        super(EncoderRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=bidirectional)

    def forward(self, input_seq, hidden = None):
        if(len(input_seq) == 0):
          return t.FloatTensor(1,input_seq.shape[1], self.hidden_size).fill_(0), None #initial h0
        
        embedded = self.embedding(input_seq) #[T,B,E]
        outputs, hidden = self.lstm(embedded, hidden) #outputs : (T, B, (2*)H)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        return outputs, hidden
      

class Attn(nn.Module):
    def __init__(self, method, hidden_size, g_size = int()):
        '''
        :param g_size:
          goal vector length (assuming 1D)
        '''
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.g_size = g_size
        self.attn = nn.Linear(self.hidden_size + g_size, hidden_size)
        self.v = nn.Parameter(t.FloatTensor(hidden_size).uniform_(0,1))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, encoder_outputs, g = None):
        '''
        :param hidden: 
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :param src_len:
            used for masking. NoneType or tensor in shape (B) indicating sequence length
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        encoder_outputs = encoder_outputs.transpose(0,1) # [B*T*H]
        attn_energies = self.score(encoder_outputs, g) # compute attention score
        
        return F.softmax(attn_energies, dim = 1).unsqueeze(1) # normalize with softmax #[B,1,T] 

    def score(self, encoder_outputs, g = None):
        if(self.g_size != 0): #g:(B,G)
          g = g.float().repeat(encoder_outputs.shape[1],1,1).transpose(1,0) #(B,G) => (T,B,G) => (B,T,G)
          inp = torch.cat((encoder_outputs, g),2) #[B,T,H] <concat> [B,T,G] ==> [B,T,(H+G)]
        else:
          inp = encoder_outputs
          
        energy = self.attn(inp)
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v, energy) # [B*1*T]
        return energy.squeeze(1) #[B*T]

class AttnDecoder(nn.Module):
    def __init__(self, layer_dims, g_size = int(), activation_fn = lambda a: a):
        '''
        :param g_size:
          goal vector length (assuming 1D)
        :param layer_dims:
            [input size + g size, first hidden layer size, ..., last hidden layer size, output size]
            input size : H
            g size : V
            output size : P(HONEMES_INVENTORY_SIZE)
        '''
        super(AttnDecoder, self).__init__()
        #self.attn = attn # [A] -> B
        
        assert(len(layer_dims) > 1)
        layer_dims[0] += g_size
        self.input_size = layer_dims[0] #[H(+G), ...]
        self.g_size = g_size
        self.output_size = layer_dims[-1] #[... ,P]
        
        #self.ff = ff # B -> C
        self.hidden_layers = nn.ModuleList([nn.Linear(layer_dims[i],layer_dims[i+1]) for i in range(len(layer_dims)-1)])
        self.activation_fn = activation_fn
        

    def forward(self, encoder_outputs, attn_weights, g = None, debug = False):
        '''
        :param word_input:
            word input for current time step, in shape (B)
        :param last_hidden:
            last hidden stat of the decoder, in shape (layers*direction*B*H)
        :param encoder_outputs:
            encoder outputs in shape (T*B*H)
        :param attn_weights:
            softmax'd attention weights from attn(encoder_outputs) in shape [B,1,T]
        :return
            decoder output (a_distro) in shape [B,P]
        
        Note: we run this one step at a time i.e. you should use a outer loop 
            to process the whole sequence
        Tip(update):
        EncoderRNN may be bidirectional or have multiple layers, so the shape of hidden states can be 
        different from that of DecoderRNN
        You may have to manually guarantee that they have the same dimension outside this function,
        e.g, select the encoder hidden state of the foward/backward pass.
        '''
        
        #attn_weights = self.attn(encoder_outputs) #[B,1,T] 
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # [B,1,T].bmm([B,T,H]) ==> (B,1,H)
        context = context.transpose(0, 1).squeeze(0)  # (B,H)
        
        if(self.g_size != 0): #g:(B,G)
          inp = torch.cat((context, g),1) #(B,H+G)
        else:
          inp = context
        
        #return self.ff(context)  #feed-forward neural net [H,H',H'', ... ,P(HONEME_INVENTORY_SIZE)]
        x = self.activation_fn(self.hidden_layers[0](inp))
        for i in range(1,len(self.hidden_layers)-1):
          x = self.activation_fn(self.hidden_layers[i](x))
        x = self.hidden_layers[-1](x)
        
        if(debug): print(x)
          
        return F.softmax(x, dim = 1) #[B,P] -- a_distro for each batch item\
        
"""
class EncoderRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers=1, dropout=0.5):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

    def forward(self, input_seq, hidden=None):
        embedded = self.embedding(input_seq)
        outputs, hidden = self.lstm(embedded, hidden)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        return outputs, hidden

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, encoder_outputs):
        '''
        :param hidden: 
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :param src_len:
            used for masking. NoneType or tensor in shape (B) indicating sequence length
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        encoder_outputs = encoder_outputs.transpose(0,1) # [B*T*H]
        attn_energies = self.score(encoder_outputs) # compute attention score
        
        return F.softmax(attn_energies).unsqueeze(1) # normalize with softmax

    def score(self, encoder_outputs):

        energy = self.attn(encoder_outputs) #[B,T,H]
        #should we apply an activation function to signals from hidden layer?
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v, energy) # [B*1*T]
        return energy.squeeze(1) #[B*T]

class AttnDecoder(nn.Module):
    def __init__(self, ff, attn):
        super(AttnDecoder, self).__init__()
        self.attn = attn # [A] -> B
        self.ff = ff # B -> C

    def forward(self, encoder_outputs):
        '''
        :param word_input:
            word input for current time step, in shape (B)
        :param last_hidden:
            last hidden stat of the decoder, in shape (layers*direction*B*H)
        :param encoder_outputs:
            encoder outputs in shape (T*B*H)
        :return
            decoder output
        Note: we run this one step at a time i.e. you should use a outer loop 
            to process the whole sequence
        Tip(update):
        EncoderRNN may be bidirectional or have multiple layers, so the shape of hidden states can be 
        different from that of DecoderRNN
        You may have to manually guarantee that they have the same dimension outside this function,
        e.g, select the encoder hidden state of the foward/backward pass.
        '''
        attn_weights = self.attn(encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,V)
        context = context.transpose(0, 1)  # (1,B,V)
        return self.ff(context)
"""