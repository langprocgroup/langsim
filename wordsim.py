#https://colab.research.google.com/drive/1jmtmD5JDRfWA3jpxjXIcsuh2uWrBr8RV

import torch
from torch import nn
if torch.cuda.is_available(): import torch.cuda as t
else: import torch as t
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import math, random
from attentionRNN import *


# parameters


PHONEME_INVENTORY_SIZE = 10 + 1
SENTINEL_ID = PHONEME_INVENTORY_SIZE - 1
VOCABULARY_SIZE = 100
MAX_WORD_LENGTH = 5

SENTINEL = object()

PHONEMES = range(PHONEME_INVENTORY_SIZE-1) #exclude sentinel

input_size = PHONEME_INVENTORY_SIZE
embed_size = 10
hidden_size = 100
g_size = VOCABULARY_SIZE
n_layers = 1 #EncoderLSTM
dropout = 0.5
learning_rate = 1e-3

batch_size = 10 #training


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
  completed = {} #stores completed sequences as they are taken out of so_far

  one_hot_dim = batch_meanings.size(1)
  batch_meaning_idx = idx_one_hot(batch_meanings)
  so_far = {g_idx:[] for g_idx in batch_meaning_idx} #stores seq as they are being generated
  
  current_batch_meanings_idx = [g_idx for g_idx in batch_meaning_idx] #updated by a == SENTINEL_ID
  current_batch_meaning = one_hot(current_batch_meanings_idx, one_hot_dim)
  
  while len(current_batch_meanings_idx) > 0:
    batch_g = t.FloatTensor(current_batch_meaning) #[B,G]
    
    input_seq = t.LongTensor([so_far[g_idx] for g_idx in current_batch_meanings_idx]).transpose(1,0) #[B,t].T ==> [t,B]

    #encoder
    outputs, _hidden = encoder(input_seq, None)

    #policy
    attn_weights = attn(outputs, batch_g)
    a_distro = attn_decoder(outputs, attn_weights, batch_g) #[B,P]
    a_distro = torch.distributions.Categorical(a_distro)
    batch_a = a_distro.sample()
    
    #update so_far & current batch idx
    new_batch_meanings_idx = []
    
    for i in range(len(current_batch_meanings_idx)):
      g_idx = current_batch_meanings_idx[i]
      a = batch_a[i]
      if a == SENTINEL_ID:
        if(completed != None):
          completed[g_idx] = so_far[g_idx] #save completed sequence
        del so_far[g_idx]
      else:
        so_far[g_idx] += [a] #update sequences excluding completed
        new_batch_meanings_idx.append(g_idx)

    current_batch_meanings_idx = new_batch_meanings_idx
    current_batch_meaning = one_hot(current_batch_meanings_idx, one_hot_dim)
    
  return completed

# Get the log probability that the agent would produce a sequence
def score_sequence(batch_meanings, batch_targets, encoder, attn, attn_decoder):
    '''
    :param batch_meanings:
      one hot encodings of meanings in shape [B,V] where V is vocabulary size
    :param batch_targets:
      batch target sequences corresponding to batch meanings in shape [B,(T)] with various sequence lengths
    
    current_batch will keep track of sequences still in production, function ends when all seqeuences have been produced
    '''
    batch_size = len(batch_meanings)
    
    scores = t.FloatTensor([0.0 for _ in range(batch_size)])
    
    t_step = 0
    
    current_batch = [i for i in range(batch_size) if t_step < len(batch_targets[i])]
    targets = [batch_targets[i][t_step] for i in current_batch] #[B]
    input_seq = [batch_targets[i][:t_step] for i in current_batch] #[B,t]
    
    
    while len(current_batch) > 0:
      batch_g = torch.stack([batch_meanings[i] for i in current_batch]) #[B,G]
      
      outputs, _hidden = encoder(t.LongTensor(input_seq).transpose(1,0), None)
      attn_weights = attn(outputs, batch_g)
      a_distro = attn_decoder(outputs, attn_weights, batch_g) #[B,P] probability distribution of phonemes
      n_log_p = -torch.log(a_distro) #cross entropy loss
      scores[current_batch] += n_log_p[range(len(current_batch)),targets] #+= [B]
      
      t_step += 1
      current_batch = [i for i in range(batch_size) if t_step < len(batch_targets[i])]
      targets = [batch_targets[i][t_step] for i in current_batch] #[B]
      input_seq = [batch_targets[i][:t_step] for i in current_batch] #[B,t]
      
    return scores

def train(vocabulary,  encoder, attn, attn_decoder, opt, batch_size=5, num_epochs=10):
    '''
    :param vocabulary:
      dict{meaning:sequence}
    :param encoder:
      EncoderRNN transforming sequence into T meaning representations (T(seq len) hidden states from RNN)
    :param attn:
      Attn 1 hidden layer feedforward neural net computing attention energies from T meaning representations
    :attn_decoder:
      Attn_decoder computes context (attention*encoder_outputs), forward into neural net to produce probability distribution of actions(phonemes)
    '''
    params = list(encoder.parameters()) + list(attn.parameters()) + list(attn_decoder.parameters())
    
    meanings = list(vocabulary.keys()) #[V]
    
    pos = 0
    for epoch in range(num_epochs):
        scores = t.FloatTensor(batch_size).fill_(0)
        opt.zero_grad()
        
        batch_meanings = one_hot(meanings[pos:min(pos+batch_size, len(meanings))], len(vocabulary)) #[B,V]
        
        pos+=batch_size
        if(pos >= len(meanings)):
          pos = 0
        
        batch_targets = [vocabulary[g_idx]+[SENTINEL_ID] for g_idx in idx_one_hot(batch_meanings)]
        
        scores = score_sequence(batch_meanings, batch_targets, encoder, attn, attn_decoder)
        loss = scores.mean()
        loss.backward()
        
        #nn.utils.clip_grad_norm_(params, 5, norm_type=2) #clip to prevent exploding gradient
        
        opt.step()
        
        if (epoch % 100 == 0):
          print(f"Epoch {epoch}, loss: {loss}")
          
    print(f"Epoch {epoch}, loss: {loss}")


def main():
    #init models
    encoder = EncoderRNN(input_size, embed_size, hidden_size, n_layers, dropout, bidirectional = True).to(device)
    attn = Attn(None, hidden_size, g_size = g_size).to(device)
    attn_decoder = AttnDecoder([hidden_size,hidden_size,PHONEME_INVENTORY_SIZE], g_size, activation_fn = torch.tanh).to(device) #should we add more layers/make them bigger?
    params = list(encoder.parameters()) + list(attn.parameters()) + list(attn_decoder.parameters())
    optimizer = torch.optim.Adam(params, lr = learning_rate, weight_decay = .0001)

    #init vocabulary
    vocabulary = generate_random_vocabulary()
    meanings = list(vocabulary.keys())

    # #sampling agent
    # batch_g_idx = [random.randint(0,len(vocabulary)-1) for _ in range(5)]
    # batch_g = one_hot(batch_g_idx, len(vocabulary))

    # target_as_str = "\n\t".join([str((i,vocabulary[i])) for i in batch_g_idx])
    # print(f"Target: \n\t{target_as_str}")

    # produced = agent_generate_word(batch_g, len(vocabulary), encoder, attn, attn_decoder, sample = True)
    # produced_as_str = "\n\t".join([str((i,t.LongTensor(produced[i]))) for i in batch_g_idx])
    # print(f"Produced: \n\t{produced_as_str}")

    #training
    import time
    n_epochs = 100


    start_time = time.time()
    train(vocabulary,  encoder, attn, attn_decoder, optimizer, batch_size=batch_size, num_epochs=n_epochs)
    end_time = time.time()

    print(f"{n_epochs} epochs in {end_time-start_time} seconds")

    #evaluation
    batch_g_idx = list(range(len(vocabulary)))
    batch_g = one_hot(batch_g_idx, len(vocabulary))

    incorrects = []
    corrects = []

    produced = agent_generate_word(batch_g, encoder, attn, attn_decoder)
    produced = {g_idx:produced[g_idx] for g_idx in batch_g_idx}

    for g_idx in batch_g_idx:
      if(len(vocabulary[g_idx]) == len(produced[g_idx]) and torch.all(torch.eq(t.FloatTensor(vocabulary[g_idx]),  t.FloatTensor(produced[g_idx])))):
        corrects.append( (g_idx, vocabulary[g_idx], t.FloatTensor(produced[g_idx])) )
      else:
        incorrects.append( (g_idx, vocabulary[g_idx], t.FloatTensor(produced[g_idx])) )
        
    print(f"Num Correct: {len(corrects)}\nNum Incorrect: {len(incorrects)}")
    #print(incorrects)


if __name__ == '__main__':
    main()

"""
PHONEME_INVENTORY_SIZE = 10
VOCABULARY_SIZE = 100
MAX_WORD_LENGTH = 5

SENTINEL = object()

PHONEMES = range(PHONEME_INVENTORY_SIZE)

def generate_random_word():
    length = random.choice(range(1, MAX_WORD_LENGTH+1))
    word = [random.choice(PHONEMES) for _ in range(length)]
    return word

def generate_random_vocabulary():
    words = [generate_random_word() for _ in range(VOCABULARY_SIZE)]
    return {i:word for i, word in enumerate(words)}

# Policy is a feedforward network that takes the memory m and goal g as input
# and outputs a softmax distribution over the following actions in A.
# Last layer is log softmax activation.

# Have the agent forward-sample a sequence
def agent_generate_word(g, policy, encoder):
    so_far = []
    while True:
        m = encoder(so_far)
        a_distro = torch.distributions.Categorical(policy(m, g))
        a = a_distro.sample()
        so_far.append(a)
        if a is SENTINEL:
            return so_far

# Get the log probability that the agent would produce a sequence
def score_sequence(g, target, policy, encoder):
    so_far = []
    score = 0.0
    for a in target:
        m = encoder(so_far)

        #we add to score because log p(a1,a2,...|m_0,g) = Σ(log p(a_i|m_0,g))
        #   but are we getting p(a_i|m_0,g) or p(a_i|m_i,g) from policy(m,g)[a]?
        #       are they equivalent?
        score += policy(m, g)[a] # log p(a|m,g) 

        so_far.append(a)
    return score

# Best loss is cross-entropy. For each true sequence, what is the log probability
# of that sequence for the policy and encoder.

def train(policy, encoder, vocabulary, batch_size=1, num_epochs=10000):
    for epoch in num_epochs:
        scores = torch.zeros(batch_size)
        opt.zero_grad()
        for i in range(batch_size):
            goal = random.choice(vocabulary)
            scores[i] = -score_sequence(goal, vocabulary[goal], policy, encoder)
        loss = scores.mean()
        loss.backward()
        opt.step()
"""
        

        
    
    
    
    
    
