import random

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

def agent_generate_word(g, policy, encoder):
    so_far = []
    while True:
        m = encoder(so_far)
        a = policy(m, g)
        so_far.append(a)
        if a is SENTINEL:
            return so_far

def punishing_loss(x, y): 
    return x == y

# Optimally the loss would be something like cross-entropy, 

def train():
    vocabulary = generate_random_vocabulary()
    policy = TODO
    encoder = TODO
    while True:
        goal = random.choice(vocubulary)
        # turn the goal into a one-hot vector
        generated = agent_generate_word(goal, policy, encoder)
        loss = punishing_loss(vocabulary[goal], generated)
        loss.backward()
        opt.step()

        
        
        

        
    
    
    
    
    
