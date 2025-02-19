import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.nn import functional as F


from scripts.Agent import Episode, iRDPGAgent, PERBuffer, generate_demonstration_episodes, collect_episode
from scripts.Env import POMDPTEnv, dt_policy, intraday_greedy_actions

from tqdm import trange, tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

df = pd.read_csv('data/train_one_month.csv')
df = df.drop(columns=['timestamp'])

def train(config, env, agent, buffer):

    actor_optim = optim.Adam(list(agent.actor_gru.parameters()) + list(agent.actor_fc.parameters()), lr=config["actor_lr"])
    critic_optim = optim.Adam(list(agent.critic_gru.parameters()) + list(agent.critic_fc.parameters()), lr=config["critic_lr"])
    

    demo_episodes = generate_demonstration_episodes(env, 
                        n_episodes=config["min_demo_episodes"])
    
    for episode in tqdm(demo_episodes, desc="Pre-filling buffer"):
        buffer.add_episode(episode)

    
    total_critic_losses = []
    total_actor_losses = []
    # Training loop
    for epoch in trange(config["epochs"], desc="Training"):

        agent_episode = collect_episode(env, agent, add_noise=True)
        buffer.add_episode(agent_episode)
        
        if len(buffer) < config["min_demo_episodes"]:
            continue
            
        batch, indices, weights = buffer.sample(config["batch_size"])
        
        critic_losses = []
        actor_losses = []
        new_priorities = []

        # Process episodes
        for episode in batch:
            obs = episode.obs.unsqueeze(0)
            actions = episode.actions.unsqueeze(0)
            rewards = episode.rewards.unsqueeze(0).unsqueeze(-1)
            expert_acts = episode.expert_actions.unsqueeze(0)
            dones = episode.dones.unsqueeze(0).unsqueeze(-1)

            with torch.no_grad():
                _, target_q, _, _ = agent.target_forward(obs)
                target_q = rewards + (1 - dones.float()) * config["gamma"] * target_q

            q_values, _ = agent.critic_forward(obs, actions)
            
            # Critic loss
            critic_loss = F.mse_loss(q_values, target_q)
            critic_losses.append(critic_loss)

            with torch.no_grad(): # Detach from graph
                expert_q, _ = agent.critic_forward(obs, expert_acts) 
                current_q = q_values.detach()

            action_probs, _, _, _ = agent(obs)
            
            # Policy gradient loss
            actor_loss = -current_q.mean()
            
            # Behavior cloning loss
            mask = (expert_q > current_q).float()
            bc_loss = F.mse_loss(action_probs, expert_acts.float()) * mask.mean()
            
            total_actor_loss = config["lambda1"] * actor_loss + config["lambda2"] * bc_loss
            actor_losses.append(total_actor_loss)

            # Update priorities
            priority = critic_loss.item() + config["lambda0"] * actor_loss.item()
            if episode.is_demo:
                priority += config["eps_demo"]
            new_priorities.append(priority)

        # Update critic
        critic_optim.zero_grad()
        critic_loss = torch.stack(critic_losses).mean()
        critic_loss.backward() 
        critic_optim.step()

        # Update actor
        actor_optim.zero_grad()
        actor_loss = torch.stack(actor_losses).mean()
        actor_loss.backward()
        actor_optim.step()

        # Update buffer priorities
        buffer.update_priorities(indices, new_priorities)
        
        # Update target networks
        agent._update_target_networks(config["tau"])

        total_critic_losses.append(critic_loss.item())
        total_actor_losses.append(actor_loss.item())

        if (epoch+1) % (config['epochs']/5) == 0:
            print(f"Epoch {epoch} | Critic Loss: {critic_loss.item():.4f} | "
                  f"Actor Loss: {actor_loss.item():.4f}")
            
            save_model(agent, filename=f"trained_irdpg_{epoch+1}.pth")

            os.makedirs("train/images", exist_ok=True)
            plt.figure()
            plt.plot(total_critic_losses, label="Critic Loss")
            plt.plot(total_actor_losses, label="Actor Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig("train/images/loss_plot.png")
            plt.close()

            

def save_model(agent, filename="trained_irdpg.pth"):
    checkpoint = {
        "actor_gru": agent.actor_gru.state_dict(),
        "actor_fc": agent.actor_fc.state_dict(),
        "critic_gru": agent.critic_gru.state_dict(),
        "critic_fc": agent.critic_fc.state_dict(),
        "target_actor_gru": agent.target_actor.state_dict(),
        "target_actor_fc": agent.target_actor_fc.state_dict(),
        "target_critic_gru": agent.target_critic.state_dict(),
        "target_critic_fc": agent.target_critic_fc.state_dict(),

        "actor_optim": agent.actor_optim.state_dict(),
        "critic_optim": agent.critic_optim.state_dict()
    }
    torch.save(checkpoint, filename)
    print(f"\nModel saved as {filename}\n")

config = {
    "epochs": 500,          # Total training epochs
    "batch_size": 32,       # Episodes per batch
    "gamma": 0.99,          # Discount factor
    "tau": 0.01,            # Target network update rate
    "lambda0": 0.6,         # Priority weight
    "lambda1": 0.8,         # Policy gradient weight
    "lambda2": 0.2,         # BC loss weight
    "actor_lr": 1e-4,       # Learning rates
    "critic_lr": 1e-3,
    "eps_demo": 0.1,        # Priority boost for demos
    "noise_std": 0.1,       # Exploration noise
    "demo_ratio": 0.3,      # Ratio of demo episodes in buffer
    "min_demo_episodes": 100, # Min demo episodes to start
    "seq_len": 60           # Match window_size
}

env = POMDPTEnv(df)
agent = iRDPGAgent(obs_dim=env.observation_space.shape[0], device=device)
buffer = PERBuffer(max_episodes=100)

train(config, env, agent, buffer)

