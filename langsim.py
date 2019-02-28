import random
import itertools

import numpy as np
import rfutils
import torch
import torch.nn as nn
import torch.nn.functional as F

EOS = "!END"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def SoftmaxFF(input_size, hidden_sizes, output_size, activation=nn.ReLU()):
        sizes = [input_size] + hidden_sizes + [output_size]
        layers = []
        for s1, s2 in rfutils.sliding(sizes, 2):
            layers.append(nn.Linear(s1, s2, bias=True))
        sequence = []
        for layer in layers[:-1]:
            sequence.append(layer)
            sequence.append(activation)
        sequence.append(layers[-1])
        sequence.append(nn.LogSoftmax())
        return nn.Sequential(*sequence)


def info_rl(P, R, rho, beta):
    # Rubin, Shamir & Tishby, p. 7, vectorized
    # S : number of states
    # A : number of actions
    # P : AxS'xS matrix giving probability of next world state
    # R : AxS matrix giving reward value of action in state
    # rho : SxA matrix giving the default policy
    # beta : single scalar value giving the penalty for control information
    # returns policy : SxA matrix giving the probability of each action in each state
    S = P.shape[-1]
    F = torch.zeros(S)
    Z = torch.zeros(S)
    while True:
        old_F = F.clone()
        # F : S
        # P @ F : AxS = sum_s' P[a,s,s'] * F[s']
        Z = (rho * torch.exp(beta * R - (P * F).sum(1))).sum(0)
        F = -torch.log(Z)
        if is_converged(old_F, F):
            break
    policy = rho/Z * torch.exp(beta * R - (P * F).sum(1))
    return policy

# Cool but we don't really need this in the world of PyTorch.
# What is the pointwise FE loss?
# R(a,s) - \beta log p(a|s)/p(a)
# To calculate this, we need a way to calculate the probabilities.
# p(a|s) comes out of the policy
# p(a) is derived from it simply...

# Alternative implementations:
# * Use batch MC estimate of the policy centroid
# * Train another network as the default policy network


# two possibly neural function
# policy : target x context -> action
# memory : context x action -> context
def actions(policy, target, memory, init, sentinel=EOS):
    context = init
    while True:
        action_p = torch.distributions.Categorical(policy(target, context))
        action = action_p.sample()
        if action is sentinel:
            break
        else:
            yield action, action_p
            context = memory(context, action)


# two neural functions
# policy : target x context_rep -> action
# encode : [action] -> context_rep
def actions_perfect_memory(policy, target, encode, sentinel=EOS):
    def cons(x, ys):
        return (x,) + ys
    def policy_with_encoding(target, context):
        return policy(target, encode(context))
    return actions(policy_with_encoding, target, sentinel, cons, ())
    
#------------------ Loss --------------------
def ib_action_loss(beta, utility, state, action, prior_p, action_p):
    cost = action_p.log_prob(action) - prior_p.log_prob(action)
    J = utility(state, action) - beta * cost
    return -J

def production_loss(loss_f, target):
    loss = 0
    for action, action_p in actions_perfect_memory(policy, target, encode):
        loss += loss_f(target, action, action_p)
    return loss




#------------ Training The Model -----------------------
def train(encoder, decoder, m_distro, optimizer, loss_fn, num_epochs, batch_size):
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        logp_m = torch.zeros(batch_size)        
        logp_m_given_u = torch.zeros(batch_size)
        logp_u_given_m = torch.zeros(batch_size)
        for i in range(batch_size):
            m = m_distro.sample()
            logp_m[i] = m_distro.log_prob(m)
            m = Variable(torch.Tensor(embed_meaning(m)))
            u_distro = torch.distributions.Categorical(encoder(m))
            u = u_distro.sample()
            logp_u_given_m[i] = u_distro.log_prob(u)
            m_hat_distro = decoder(u)
            logp_m_given_u[i] = m_hat_distro[m.item()]
        loss = -torch.mean(torch.sum(torch.exp(logp_u_given_m+logp_m) * logp_m_given_u)) # cond ent
        loss.backward()
        optimizer.step()

def parity_data(dim, k):
    input = [random.choice(range(k)) for _ in range(dim)]
    output = sum(input) % 2 # 0 if even, 1 if odd
    return input, output

def simple_data():
    k = [random.choice(range(10)) for _ in range(10)]
    return k, int(k[0]>5)

def two_in_data():
    k = [random.choice(range(10)) for _ in range(10)]
    return k, int(2 in k)

def xor_data():
    k = [random.choice(range(2)) for _ in range(2)]
    return k, int((k[0] or k[1]) and not (k[0] and k[1]))

def addition():
    k = [random.choice(range(5)) for _ in range(2)]
    return k, sum(k)

def terrible():
    a, b, c, d, e = [random.choice(range(1000)) for _ in range(5)]
    x = 2 * (a + b)
    y = b % (c+1)
    z = d*e+1
    return [a,b,c,d,e], a * (x+y+z) % 10

def mt():
    k = 10000
    random.seed()
    a = random.choice(range(k))
    random.seed(a)
    input = [0]*k
    input[a] = 1
    return input, random.choice(range(k))


# WOrks better for negative beta!!! ???? !!! ???
def autoencode(ff, data, beta=1, batch_size=1, **hyperparams):
    opt = torch.optim.Adam(ff.parameters(), **hyperparams)
    while True:
        opt.zero_grad()
        m = [data() for _ in range(batch_size)] # BxM
        logp_u_given_m = ff.forward(torch.Tensor(m)) # BxU
        logp_u = torch.log(logp_u_given_m.exp().mean(0))
        # minimize H[U|M] to get a deterministic function of m...
        condent = torch.distributions.Categorical(logp_u_given_m.exp()).entropy().mean()
        ent = torch.distributions.Categorical(logp_u.exp()).entropy().mean()
        mi = ent - condent
        J = mi - beta*ent
        (-J).backward()
        opt.step()
        print(" I[L:M] =", mi.item(), "\t H[L] =", ent.item(), end="\r")

# Discovery: You have to promote entropy in order to keep the language from collapsing.
# At least for 10 -> 10.
# Suppose we have 10 -> 100? Then entropy should be bad right? Yeah, you'd get synonyms.

# Objective: Make H[L|M] low but H[L] high.
# That gives high MI, because a pooling equilibrium will have lower H[L].
# min J = H[L|M] + beta*H[L] for positive beta

# Discovery: High beta is bad and makes it get caught in bad equilibria,
# presumably because exploration requires high entropy...
# what if we make entropy a plus? (negative beta)
# Wow, making entropy a plus makes things much better! Wow

# Suppose we minimize H[L|M] - beta*H[L]
# = H[L|M] - (beta-1)H[L] - H[L]
# = -(beta-1)H[L] - I[L:M]
# = max I[L:M] + (beta-1)H[L]
# Why is entropy a plus??????????

# Idea: Entropy is easy to minimize (!), so if it is not a plus, it overpowers the other terms.
# Adding an entropy objective is like sampling at higher temperature.





def indices(k):
    onehot = [0]*k
    onehot[random.choice(range(k))] = 1
    return onehot

# encoder : a -> b
# decoder : b -> a
# data : () -> a
def autoencoder(encoder, decoder, data, batch_size=1, **hyperparams):
    params = itertools.chain(encoder.parameters(), decoder.parameters())
    opt = torch.optim.Adam(params, **hyperparams)
    while True:
        opt.zero_grad()
        m = [data() for _ in range(batch_size)] # BxX 
        u = encoder.forward(torch.Tensor(m)) # 
        mhat = decoder.forward(u)
        loss = mhat.log_prob(m)
        loss.backward()
        opt.step()
        print(loss.item())
    

def simple_train(ff, data, batch_size=1, **params):
    opt = torch.optim.Adam(ff.parameters(), **params)
    while True:
        opt.zero_grad()
        xy = [data() for _ in range(batch_size)]
        x, y = zip(*xy)
        yhat = ff.forward(torch.Tensor(x))
        loss = F.nll_loss(yhat, torch.tensor(y))
        loss.backward()
        opt.step()
        print(loss.item())

# Alternate FE training idea: create two networks,
# a context-sensitive policy network and a context-insensitive default network.
# The FE penalty is the difference between the two.
# ff : X x C -> Y
# ff0 : X -> Y
# ie, information from X is kostenfrei, but information from C is costly
def double_network_fe_train(ff, ff0, data, A, beta, batch_size=10000, **hparams):
    params = itertools.chain(ff.params(), ff0.params())
    opt = torch.optim.Adam(params, **hparams)
    while True:
        opt.zero_grad()
        
        xcy = [data() for _ in range(batch_size)]
        x, y = zip(*xy)
         
        yhat = ff.forward(torch.Tensor(x), torch.Tensor(c))
        yhat0 = ff0.forward()
        
        h_asc = F.nll_loss(yhat, torch.tensor(y), reduce=False) # -log p(a|s,c)
        h_as = F.nll_loss(yhat0, torch.tensor(y), reduce=False) # -log p(a|s)
        cost = h_as - h_asc
        loss = utility - beta*cost
        
        (-loss).backward()
        opt.step()
        print(loss.item())

def fe_train(ff, data, A, beta, batch_size=10000, **hyperparams):
    default_policy = torch.zeros(A)
    opt = torch.optim.Adam(ff.parameters(), **hyperparams)
    while True:
        # Normalize the default policy
        default_policy = default_policy - default_policy.logsumexp(0)
        print(default_policy.exp())

        # Training step
        opt.zero_grad()
        xy = [data() for _ in range(batch_size)]
        x, y = zip(*xy)
        yhat = ff.forward(torch.Tensor(x))
        a = torch.distributions.Categorical(torch.exp(yhat)).sample()
        value = utility(a, y)

        # Calculation of cost and reward
        reward = F.nll_loss(yhat, torch.tensor(y), reduce=False) # - log p(a|s)

        cost = yhat - default_policy # control cost of the overall policy wrt the samples
        local_cost = cost[:, yhat.argmax(1)].mean(0) # cost for each action actually evaluated
        
        loss = (reward - beta*local_cost).mean()

        # Optimzation
        loss.backward(retain_graph=True)
        opt.step()
        print(reward.mean().item())
        default_policy = yhat.logsumexp(0) # Unnormalized MC estimate of policy centroid
        
# K-dimensional loss...useful?
def main():
    policy = SoftmaxFF(input_size, output_size).to(device)
    loss_fn = cross_entropy_loss #this can be switched with torch.nn.MSELoss
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = .0001)
    train(model, optimizer, loss_fn, training_data, num_epochs=NUM_EPOCHS)

if __name__ == '__main__':
    main()
