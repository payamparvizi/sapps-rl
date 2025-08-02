import os
import torch
from torch.distributions import Normal, Independent
import numpy as np
from scipy.fftpack import fft
import pickle
import random
import gymnasium as gym


def compute_sm(task, policy, length=int(1000), episodes=int(2)):

    action_index = int(0)
    deterministic = True
    
    env = gym.make(task)
    
    rewards, actionss, eps_lens = run_actions(env, policy, action_index,
                                              length, episodes, deterministic)
    freqs, amplitudes = fourier_from_actions(actionss, eps_lens)
    sm = smoothness_score(amplitudes)
    return sm

def smoothness_score(amplitudes):
    smoothness_score = np.mean(amplitudes * normalized_freqs(amplitudes))
    return smoothness_score

def normalized_freqs(amplitudes):
    return np.linspace(0, 1, amplitudes.shape[0])

def fourier_from_actions(actionss, ep_lens):
    fouriers = list(map(fourier_transform, cut_data(actionss, ep_lens)))
    return combine(fouriers)


def cut_data(actionss, ep_lens):
    median = int(np.median(ep_lens))
    # print("median:", median)
    same_len = map(lambda x: x[:median], filter(lambda x: len(x) >= median, actionss))
    return same_len


def fourier_transform(actions, T=0.002):
    N = len(actions)
    # print(N)
    x = np.linspace(0.0, N*T, N)
    y = actions
    yf = fft(y)
    freq = np.linspace(0.0, 1.0/(2.0*T), N//2)
    amplitudes = 2.0/N * np.abs(yf[0:N//2])
    # plt.plot(freq, amplitudes)
    return freq, amplitudes


def combine(fouriers):
    freqs = fouriers[0][0]
    amplitudess = np.array(list(map(lambda x: x[1], fouriers)))
    amplitudes = np.mean(amplitudess, axis=0)
    return freqs, amplitudes


def action_sampling(policy, obs):
    obs = obs.reshape(1, obs.size)
    
    (mean, std), _ = policy.actor(obs)
    P = Normal(mean, std)
    P = Independent(P, 1) 
    act = P.sample()
    log_prob = P.log_prob(act)
    prob = log_prob.exp()
    act = act.squeeze().cpu().numpy()
    return mean, act, prob


def run_actions(env, policy, action_index=0, forced_ep_len=None, num_episodes=10, deterministic=False):

    r, d, ep_ret, ep_len, n = 0, False, 0, 0, 0
    o, _  = env.reset()
    
    actionss = []
    ep_lens = []
    actions = []
    amplitudess = []
    rs = []
    r_sum = 0
    while n < num_episodes:
    
        _, a, _ = action_sampling(policy, o)
        actions.append(a[action_index])
        o, r, d, _, _ = env.step(a)
        r_sum += r
        ep_len += 1
        
        if (forced_ep_len == None and d) or (ep_len == forced_ep_len):
            ep_lens.append(ep_len)
            actionss.append(actions)
            actions = []
            r, d, ep_ret, ep_len = 0, False, 0, 0
            o, _ = env.reset()
            rs.append(r_sum)
            r_sum = 0
            n += 1
            # print('Number of testing Episodes: ', n)
        
    ep_lens = np.array(ep_lens)
    return rs, actionss, ep_lens


def run_policy(env, policy, max_ep_len=None, num_episodes=100, render=True):
    r, d, ep_ret, ep_len, n = 0, False, 0, 0, 0 
    o, _ = env.reset()
    while n < num_episodes:
        _, a, _ = action_sampling(policy, o)
        o, r, d, _, _ = env.step(a)
        ep_ret += r
        ep_len += 1
    
        if d or (ep_len == max_ep_len):
            # print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            r, d, ep_ret, ep_len = 0, False, 0, 0
            o, _ = env.reset()
            n += 1
    





