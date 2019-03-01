import random

import torch

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
        score = policy(m, g)[a] # log p(a|m,g)
        so_far.append(v)
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
        
        

        
    
    
    
    
    
