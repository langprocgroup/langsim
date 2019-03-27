import torch

if torch.cuda.is_available():
    import torch.cuda as t
else:
    import torch as t
    
from torch import nn
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
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, dropout=dropout, bidirectional=bidirectional)

    def forward(self, input_seq, hidden = None):
        if(len(input_seq) == 0):
          return t.FloatTensor(1,input_seq.shape[1], self.hidden_size).fill_(0), None #initial h0
        
        embedded = self.embedding(input_seq) #[T,B,E]
        outputs, hidden = self.lstm(embedded, hidden) #outputs : (T, B, (2*)H)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        return outputs, hidden
      

class Attn(nn.Module):
    def __init__(self, hidden_size, g_size = int()):
        '''
        :param g_size:
          goal vector length (assuming 1D)
        '''
        super(Attn, self).__init__()
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
          
        return F.softmax(x, dim = 1) #[B,P] -- a_distro for each batch item

#Helper functions

def one_hot(integers, k):
  to_return = t.FloatTensor(len(integers),k).fill_(0)
  to_return[range(len(integers)), integers] = 1
  return to_return

def idx_one_hot(one_hots, dim = 1):
  '''
  :param one_hots:
    batch 1 hots in shape of [B,K]
  '''
  
  return [t.item() for t in torch.argmax(one_hots, dim)]




def generate_random_word():
    length = random.choice(range(1, MAX_WORD_LENGTH+1))
    word = [random.choice(PHONEMES) for _ in range(length)]
    return word

def generate_random_vocabulary():
    words = [generate_random_word() for _ in range(VOCABULARY_SIZE)]
    return {i:word for i, word in enumerate(words)}

  
  
  # Have the agent forward-sample a sequence
def agent_generate_word(batch_meanings, encoder, attn, attn_decoder):
  '''
  :param batch_meanings:
      one hot encodings of meanings in shape [B,V] where V is vocabulary size
  '''
  #can you read one_of_k_length from batch_meanings
  
  completed = {} #stores completed sequences as they are taken out of so_far
  
  dim_one_hot = batch_meanings.size(1)
  batch_meaning_idx = idx_one_hot(batch_meanings)
  so_far = {g_idx:[] for g_idx in batch_meaning_idx} #stores seq as they are being generated
  
  sequence_probs = {g_idx:1 for g_idx in batch_meaning_idx}
  
  current_batch_meanings_idx = [g_idx for g_idx in batch_meaning_idx] #updated by a == SENTINEL_ID
  current_batch_meaning = one_hot(current_batch_meanings_idx, dim_one_hot)
  
  while len(current_batch_meanings_idx) > 0:
    batch_g = t.FloatTensor(current_batch_meaning) #[B,G]
    input_seq = t.LongTensor([so_far[g_idx] for g_idx in current_batch_meanings_idx]).transpose(1,0) #[B,t].T ==> [t,B]

    #encoder
    outputs, _hidden = encoder(input_seq, None)

    #policy
    attn_weights = attn(outputs, batch_g)
    a_distro = attn_decoder(outputs, attn_weights, batch_g) #[B,P]
    batch_a = torch.distributions.Categorical(a_distro).sample() #[B]
    
    for i in range(len(batch_a)):
      sequence_probs[current_batch_meanings_idx[i]] *= a_distro[i][batch_a[i]]
      
    #update so_far & current batch idx
    new_batch_meanings_idx = []
    
    for i in range(len(current_batch_meanings_idx)):
      g_idx = current_batch_meanings_idx[i]
      a = batch_a[i]
      if a == SENTINEL_ID:
        if(completed != None):
          completed[g_idx] = t.LongTensor(so_far[g_idx]) #save completed sequence
        del so_far[g_idx]
      else:
        so_far[g_idx] += [a] #update sequences excluding completed
        new_batch_meanings_idx.append(g_idx)
        
    #update current_batch_meanings
    current_batch_meanings_idx = new_batch_meanings_idx
    current_batch_meaning = one_hot(current_batch_meanings_idx, dim_one_hot)
    
  return completed, sequence_probs

    
def listener(utterance, encoder, attn, attn_decoder):
  '''
  :param utterance:
    sequence produced by agent conditioned on some goal
    shape: [T]
    
  :return:
    decoded meanings from utterances
  '''
  utterance = utterance.view(-1,1) #[T,B=1]
  outputs, _ = encoder(utterance) #o: (T, B = 1, H)
  attn_weights = attn(outputs)
  g_distro = attn_decoder(outputs, attn_weights) #[B,G]
  
  return g_distro #predicted goal


def train_joint(meanings, 
                a_encoder, a_attn, a_attn_decoder,
                d_encoder, d_attn, d_attn_decoder,
                opt, batch_size = 4, num_epochs=10):
  
  #meanings should be 1 hots
  
  loss = 0
  for epoch in range(num_epochs):
    
    
    pos = epoch%len(meanings)
    batch_meanings = meanings[pos:pos+1] #[1,VOCABULARY_SIZE]
    
    produced, seq_probs = agent_generate_word(batch_meanings, a_encoder, a_attn, a_attn_decoder)
    #produced: dict w/ meaning g as key and produced sequence as value 
    #seq_probs: dict w/ meaning g as key and probability of sequence produced as value
    
    g = list(produced.keys())[0]
    g_distro = listener(produced[g], d_encoder, d_attn, d_attn_decoder).squeeze(0) #[G]
    loss -= seq_probs[g] * torch.log(g_distro)[g]
      
    
    if(epoch%batch_size == 0):
      print(f"Epoch {epoch}, loss: {loss/batch_size}, seq_prob: {sum(seq_probs.values())/len(seq_probs)}")
      
      loss.backward()
      opt.step()
      
      opt.zero_grad()
      loss = 0
    
  print(f"Epoch {epoch}, loss: {loss}")
  

if __name__ == "__main__":
	PHONEME_INVENTORY_SIZE = 10 + 1
	SENTINEL_ID = PHONEME_INVENTORY_SIZE - 1
	VOCABULARY_SIZE = 100
	MAX_WORD_LENGTH = 5
	PHONEMES = range(PHONEME_INVENTORY_SIZE-1) #exclude sentinel

	input_size = PHONEME_INVENTORY_SIZE
	embed_size = PHONEME_INVENTORY_SIZE
	hidden_size = 100
	g_size = VOCABULARY_SIZE
	n_layers = 1 #EncoderLSTM
	dropout = 0.5
	learning_rate = 1e-3

	batch_size = 20 #training

	#init models

	#agent: g -> encoderRNN -> h -> attn -> c -> attn_decoder -> a -- ... --> seq
	agent_encoder = EncoderRNN(input_size, embed_size, hidden_size, n_layers, dropout, bidirectional = True).to(device)
	agent_attn = Attn(hidden_size, g_size = g_size).to(device)
	agent_attn_decoder = AttnDecoder([hidden_size,hidden_size,PHONEME_INVENTORY_SIZE], g_size, activation_fn = torch.tanh).to(device)
	#model

	#listener: seq -> encoderRNN -> h -> attn -> c -> attn_decoder -> g
	listener_encoder = EncoderRNN(input_size, embed_size, hidden_size, n_layers, dropout, bidirectional = True).to(device)
	listener_attn = Attn(hidden_size).to(device)
	listener_attn_decoder = AttnDecoder([hidden_size, hidden_size, VOCABULARY_SIZE], g_size = 0, activation_fn = torch.tanh).to(device)

	params = list(agent_encoder.parameters()) + list(agent_attn.parameters()) + list(agent_attn_decoder.parameters())
	params += list(listener_encoder.parameters()) + list(listener_attn.parameters()) + list(listener_attn_decoder.parameters())
	optimizer = torch.optim.Adam(params, lr = learning_rate, weight_decay = .0001)




	#init meanings
	meanings = list(range(VOCABULARY_SIZE))


	#train
	batch_g = one_hot(meanings, VOCABULARY_SIZE)
	train_joint(batch_g, 
	            agent_encoder, agent_attn, agent_attn_decoder,
	            listener_encoder, listener_attn, listener_attn_decoder,
	            optimizer, batch_size = batch_size, num_epochs = 100)

	#eval (e.g. on first 20 meanings)
	batch_g = one_hot(meanings[:20], VOCABULARY_SIZE)
	produced, _ = agent_generate_word(batch_g, agent_encoder, agent_attn, agent_attn_decoder)

	for g in range(20):
	  g_distro = listener(produced[g], listener_encoder, listener_attn, listener_attn_decoder).squeeze(0) #[G]
	  print(f"\nu|g={g} : {t.LongTensor(produced[g])}")
	  print(f"p(g|u) : {g_distro[g]}\n")


