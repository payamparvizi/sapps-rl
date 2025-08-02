#!/usr/bin/env python3

import argparse
import datetime
import os
import pprint
import pickle

import numpy as np
import torch
# from mujoco_env import make_mujoco_env
import gymnasium as gym
from torch import nn
from torch.distributions import Distribution, Independent, Normal
from torch.optim.lr_scheduler import LambdaLR

from packages.tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
# from tianshou.highlevel.logger import LoggerFactoryDefault
from packages.tianshou.policy import PPOPolicy
from packages.tianshou.policy.base import BasePolicy
from packages.tianshou.trainer import OnpolicyTrainer
from packages.tianshou.utils.net.common import ActorCritic, Net
from packages.tianshou.utils.net.continuous import ActorProb, Critic
from packages.tianshou.env import DummyVectorEnv
from packages.tianshou.utils import WandbLogger
from torch.utils.tensorboard import SummaryWriter
from utils.arguments.arguments import get_args
from utils.compute_smoothness import compute_sm


def test_ppo(args: argparse.Namespace = get_args()) -> None:

    env = gym.make(args.task)
    train_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.test_num)])

    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    args.hidden_sizes = [args.hidden_size, args.hidden_size]
    
    # model
    net_a = Net(
        args.state_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
        device=args.device,
    )
    actor = ActorProb(
        net_a,
        args.action_shape,
        unbounded=True,
        device=args.device,
    ).to(args.device)
    net_c = Net(
        args.state_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
        device=args.device,
    )
    critic = Critic(net_c, device=args.device).to(args.device)
    actor_critic = ActorCritic(actor, critic)

    torch.nn.init.constant_(actor.sigma_param, -0.5)
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)

    for m in actor.mu.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)

    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

    lr_scheduler = None
    if args.lr_decay:
        # decay learning rate to 0 linearly
        max_update_num = np.ceil(args.step_per_epoch / args.step_per_collect) * args.epoch

        lr_scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num)

    def dist(*logits):
        return Independent(Normal(*logits), 1)

    if args.regularization_case == "standard_PPO":
        args.ar_case = 0
    elif args.regularization_case == "PPO_CAPS":
        args.ar_case = 1
    elif args.regularization_case == "PPO_APS":
        args.ar_case = 2

    policy: PPOPolicy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        reward_normalization=args.rew_norm,
        action_scaling=True,
        action_bound_method=args.bound_action_method,
        lr_scheduler=lr_scheduler,
        action_space=env.action_space,
        eps_clip=args.eps_clip,
        value_clip=args.value_clip,
        dual_clip=args.dual_clip,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
        ar_case=args.ar_case,
        lambda_T=args.lambda_T,
        lambda_S=args.lambda_S,
        sigma_s_bar=args.sigma_s_bar,
        c_homog = args.c_homog,
        lambda_P = args.lambda_P,
        noise_pym = args.noise_pym
    )

    # load a previous policy
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=args.device)
        policy.load_state_dict(ckpt["model"])
        train_envs.set_obs_rms(ckpt["obs_rms"])
        test_envs.set_obs_rms(ckpt["obs_rms"])
        print("Loaded agent from: ", args.resume_path)

    # collector
    buffer: VectorReplayBuffer | ReplayBuffer
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(args.buffer_size)
        
    train_collector = Collector(policy=policy, env=train_envs, buffer=buffer, 
                                exploration_noise=True, task_sm=args.task, 
                                length_sm=args.length_sm, episodes_sm=args.episodes_sm)
    
    test_collector = Collector(policy=policy, env=test_envs, task_sm=args.task, 
                               length_sm=args.length_sm, episodes_sm=args.episodes_sm)

    
    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "ppo"
    log_name = os.path.join(args.task, args.algo_name, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)

    logger = WandbLogger(
    save_interval= 1,
    train_interval = 2,
    test_interval = 1,
    update_interval = 2,
    
    name=log_name.replace(os.path.sep, "_"),
    config=args,
    project="results_mujoco"
    )
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger.load(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.torch"))

    if not args.watch:
        # trainer
        result = OnpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            repeat_per_collect=args.repeat_per_collect,
            episode_per_test=args.test_num,
            batch_size=args.batch_size,
            step_per_collect=args.step_per_collect,
            # save_best_fn=save_best_fn,
            logger=logger,
            test_in_train=False,
        ).run()
        pprint.pprint(result)

    # Let's watch its performance!
    test_envs.seed(args.seed)
    test_collector.reset()
    collector_stats = test_collector.collect(n_episode=args.test_num, render=args.render)
    print(collector_stats)

    policy_file_path = os.path.join(log_path, "policy.torch")
    policy.load_state_dict(torch.load(policy_file_path))    
    
if __name__ == "__main__":
    test_ppo(get_args())
