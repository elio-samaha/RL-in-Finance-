import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.nn import functional as F


from scripts.Agent import iRDPGAgent, PERBuffer, generate_demonstration_episodes, collect_episode
from scripts.Env import POMDPTEnv

from tqdm import trange, tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

df = pd.read_csv('data/train_one_month.csv')

def train(config, env, agent, buffer):

    # Save model folder
    run_folder = get_model_folder()

    actor_params = list(agent.actor_gru.parameters()) + list(agent.actor_fc.parameters())
    critic_params = list(agent.critic_gru.parameters()) + list(agent.critic_fc.parameters())

    actor_optim = optim.Adam(actor_params, lr=config["actor_lr"])
    critic_optim = optim.Adam(critic_params, lr=config["critic_lr"])
    

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
        for i, episode in enumerate(batch):

            # Reset hidden states
            h_actor, h_critic = None, None

            obs = episode.obs.unsqueeze(0)
            actions = episode.actions.unsqueeze(0)
            rewards = episode.rewards.unsqueeze(0).unsqueeze(-1)
            expert_acts = episode.expert_actions.unsqueeze(0)
            dones = episode.dones.unsqueeze(0).unsqueeze(-1)

            ep_critic_loss = 0.0
            ep_actor_loss = 0.0

            T = len(obs)
            for t in range(T):

                ob_t = obs[:, t, :]
                action_t = actions[:, t, :]
                reward_t = rewards[:, t, :]
                expert_act_t = expert_acts[:, t, :]
                done_t = dones[:, t, :]

                if t < T-1:
                    ob_next = obs[:, t+1, :]
                else:
                    ob_next = torch.zeros_like(ob_t)

                with torch.no_grad():
                    _, target_q_t, _, _ = agent.target_forward(ob_next)
                    target_q_t = reward_t + (1 - done_t.float()) * config["gamma"] * target_q_t

                q_values_t, h_critic = agent.critic_forward(ob_t, action_t, h_critic)

                # Critic loss
                critic_loss_t = F.mse_loss(q_values_t, target_q_t, reduction="none")
                critic_loss_t = (critic_loss_t * weights[i]).mean()

                # Action loss
                action_probs_t, h_actor = agent(ob_t, h_actor)
                action_one_hot_t = F.gumbel_softmax(action_probs_t, tau=1, hard=True) # Ensure differentiability

                current_q_t = q_values_t.detach()

                actor_loss_t = -current_q_t.mean()

                if not episode.is_demo:
                    # Behavior cloning loss
                    with torch.no_grad():
                        expert_q_t, _ = agent.critic_forward(ob_t, expert_act_t, h_critic)
                    mask_t = (expert_q_t > current_q_t).float()
                    bc_loss_t = (F.mse_loss(action_one_hot_t, expert_act_t.float()) * mask_t).mean()
                else:
                    bc_loss_t = torch.tensor(0.0, device=device)

                total_actor_loss_t = config["lambda1"] * actor_loss_t + config["lambda2"] * bc_loss_t

                # Update episode losses
                ep_critic_loss += critic_loss_t
                ep_actor_loss += total_actor_loss_t
                
            # Normalize episode losses
            ep_critic_loss /= T
            ep_actor_loss /= T

            # Store episode losses
            critic_losses.append(ep_critic_loss)
            actor_losses.append(ep_actor_loss)

        # Update critic
        critic_optim.zero_grad()
        critic_loss = torch.stack(critic_losses).mean()
        critic_loss.backward() 
        critic_optim.step()

        # Update actor
        actor_optim.zero_grad()
        actor_loss = torch.stack(actor_losses).mean()
        actor_loss.backward()

        actor_grad_norm = torch.norm(torch.cat([p.grad.view(-1) for p in actor_params])) # Gradient norm

        actor_optim.step()

        # Update priorities
        for i, episode in enumerate(batch):
            ep_loss = critic_losses[i].detach().item()
            ep_priority = actor_grad_norm.item() * config['lambda0'] + ep_loss

            if episode.is_demo:
                ep_priority += config["eps_demo"]
            new_priorities.append(ep_priority)

        # Update buffer priorities
        buffer.update_priorities(indices, new_priorities)
        
        # Update target networks
        agent._update_target_networks(config["tau"])

        total_critic_losses.append(critic_loss.item())
        total_actor_losses.append(actor_loss.item())

        if (epoch+1) % (config['epochs']/5) == 0:
            print(f"Epoch {epoch+1} | Critic Loss: {critic_loss.item():.4f} | "
                  f"Actor Loss: {actor_loss.item():.4f}")
            
            save_model(agent, filename=f"trained_irdpg_{epoch+1}.pth", run_folder=run_folder)

    os.makedirs("images/train", exist_ok=True)
    plt.figure()
    plt.plot(total_critic_losses, label="Critic Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("images/train/critic_loss_plot.png")
    plt.close()

    plt.figure()
    plt.plot(total_actor_losses, label="Actor Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("images/train/actor_loss_plot.png")
    plt.close()

            

def save_model(agent, filename="trained_irdpg.pth", run_folder=""):
    path_dir = "models/" + run_folder
    os.makedirs(path_dir, exist_ok=True)

    checkpoint = {
        "actor_gru": agent.actor_gru.state_dict(),
        "actor_fc": agent.actor_fc.state_dict(),
        "critic_gru": agent.critic_gru.state_dict(),
        "critic_fc": agent.critic_fc.state_dict(),
        "target_actor_gru": agent.target_actor.state_dict(),
        "target_actor_fc": agent.target_actor_fc.state_dict(),
        "target_critic_gru": agent.target_critic.state_dict(),
        "target_critic_fc": agent.target_critic_fc.state_dict(),
    }
    torch.save(checkpoint, path_dir+filename)
    print(f"\nModel saved as {filename}\n")

def get_model_folder(base_dir="models/"):
    os.makedirs(base_dir, exist_ok=True)
    existing_runs = sorted(
            [d for d in os.listdir(base_dir) if d.isdigit()],
            key=lambda x: int(x))
    next_run_number = 1 if not existing_runs else int(existing_runs[-1]) + 1
    run_folder = os.path.join(base_dir, f"{next_run_number:02d}") 
    os.makedirs(run_folder, exist_ok=True)

    return run_folder

config = {
    "epochs": 500,          # Total training epochs
    "batch_size": 32,       # Episodes per batch
    "gamma": 0.99,          # Discount factor
    "tau": 0.001,            # Target network update rate
    "lambda0": 0.6,         # Priority weight
    "lambda1": 0.8,         # Policy gradient weight
    "lambda2": 0.2,         # BC loss weight
    "actor_lr": 1e-4,       # Learning rates
    "critic_lr": 1e-3,
    "eps_demo": 0.1,        # Priority boost for demos
    "noise_std": 0.1,       # Exploration noise
    "min_demo_episodes": 50, # Min demo episodes to start
    "seq_len": 3          # Match window_size
}

env = POMDPTEnv(df, window_size=config["seq_len"])
agent = iRDPGAgent(obs_dim=env.observation_space.shape[0], device=device)
buffer = PERBuffer(max_episodes=75)

train(config, env, agent, buffer)

