import os

import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.nn import functional as F

from scripts.Agent import iRDPGAgent
from scripts.PERBuffer import PERBuffer, generate_demonstration_episodes, collect_episode
from scripts.Env import POMDPTEnv
from scripts.utils import save_model, get_model_folder

from tqdm import trange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

df = pd.read_csv('data/train_one_month.csv')

def train(config, env, agent, buffer, model_folder=None):

    # Save model folder
    run_folder = get_model_folder(folder_name=model_folder)

    actor_params = list(agent.actor_gru.parameters()) + list(agent.actor_fc.parameters())
    critic_params = list(agent.critic_gru.parameters()) + list(agent.critic_fc.parameters())

    actor_optim = optim.Adam(actor_params, lr=config["actor_lr"])
    critic_optim = optim.Adam(critic_params, lr=config["critic_lr"])
    

    demo_episodes = generate_demonstration_episodes(env, 
                        n_episodes=config["min_demo_episodes"])
    
    for episode in demo_episodes:
        buffer.add_episode(episode)

    
    total_critic_losses = []
    total_actor_losses = []
    # Training loop
    for epoch in trange(config["epochs"], desc="Training"):

        agent_episode = collect_episode(env, agent)
        buffer.add_episode(agent_episode)
        
        if len(buffer) < config["min_demo_episodes"]:
            continue
            
        batch, indices, weights = buffer.sample(config["batch_size"])
        
        critic_losses = []
        actor_losses = []
        new_priorities = []
        actor_gradients = []

        # Process episodes
        for i, episode in enumerate(batch):

            # Reset hidden states
            h_actor, h_critic, h_t = None, None, None

            obs = episode.obs # [T, obs_dim]
            actions = episode.actions # [T, action_dim]
            rewards = episode.rewards # [T]
            expert_acts = episode.expert_actions # [T, action_dim]
            dones = episode.dones # [T]


            ep_critic_loss = 0.0
            ep_actor_loss = 0.0

            T = len(obs)
            for t in range(T):

                ob_t = obs[t].unsqueeze(0)
                action_t = actions[t].unsqueeze(0)
                action_prev = actions[t-1].unsqueeze(0) if t > 0 else torch.zeros_like(action_t)
                reward_t = rewards[t]
                expert_act_t = expert_acts[t].unsqueeze(0)
                done_t = dones[t]

                if t < T-1:
                    ob_next = obs[t+1].unsqueeze(0)
                else:
                    ob_next = torch.zeros_like(ob_t)

                # Hidden state ebedding
                encoder_input = torch.concat([ob_t, action_prev], dim=-1)
                h_t = agent.encode(encoder_input, h_t)

                # --Target Q--
                with torch.no_grad():
                    next_encoder_input = torch.concat([ob_next, action_t], dim=-1)
                    h_next = agent.encode(next_encoder_input, h_t)

                    _, target_q_t, _, _ = agent.target_forward(h_next)
                    target_q_t = reward_t + (1 - done_t.float()) * config["gamma"] * target_q_t

                # --Actor forward--
                action_probs_t, h_actor = agent.actor_forward(h_t, h_actor)
                # Add noise
                noise = torch.normal(0, config["noise_std"], size=action_probs_t.shape).to(device)
                action_probs_t = (action_probs_t + noise).requires_grad_()

                # --Critic forward--
                q_values_t, h_critic = agent.critic_forward(
                    h_t, 
                    action_probs_t,
                    h_critic)

                # Compute ∇_a Q
                action_copy = action_probs_t.clone().detach().requires_grad_()
                q_values_copy, _ = agent.critic_forward(h_t.clone().detach(), action_copy, h_critic.clone().detach())
                grad_wrt_action = torch.autograd.grad(
                    outputs=q_values_copy.mean(),
                    inputs=action_copy,
                    retain_graph=False,  
                    create_graph=False,
                )[0]
                actor_grad_t = grad_wrt_action.abs().mean()
                actor_grad_t = torch.nan_to_num(actor_grad_t, nan=0.0)
                actor_gradients.append(actor_grad_t.item())

                # --Critic loss--
                critic_loss_t = F.mse_loss(q_values_t, target_q_t, reduction="none")
                critic_loss_t = (critic_loss_t * weights[i]).mean()

                #with torch.no_grad():
                current_q_t = q_values_t

                # --Actor loss--
                actor_loss_t = -current_q_t.mean()

                if not torch.isnan(expert_act_t).any():
                    with torch.no_grad():
                        expert_q_t, _ = agent.critic_forward(h_t, expert_act_t.unsqueeze(0))

                    mask_t = (expert_q_t > current_q_t).float()
                    bc_loss_t = (F.mse_loss(action_probs_t.squeeze(0), expert_act_t.float(), reduction="none") * mask_t).mean()

                    total_actor_loss_t = config["lambda1"] * actor_loss_t + config["lambda2"] * bc_loss_t
                else:
                    total_actor_loss_t = config["lambda1"] * actor_loss_t

                # Update episode losses
                ep_critic_loss += critic_loss_t
                ep_actor_loss += total_actor_loss_t

                if done_t:
                    break
                
            # Normalize episode losses
            ep_critic_loss = ep_critic_loss / T
            ep_actor_loss = ep_actor_loss / T

            # Store episode losses
            critic_losses.append(ep_critic_loss)
            actor_losses.append(ep_actor_loss)

        # # Update critic
        # critic_optim.zero_grad()
        # critic_loss = torch.stack(critic_losses).mean()
        # critic_loss.backward(retain_graph=True) 
        # critic_optim.step()

        # # Update actor

        # actor_optim.zero_grad()
        # actor_loss = torch.stack(actor_losses).mean()
        # actor_loss.backward()
        # actor_optim.step()

        # -- Update both networks --
        critic_optim.zero_grad()
        actor_optim.zero_grad()

        critic_loss = torch.stack(critic_losses).mean()
        actor_loss = torch.stack(actor_losses).mean()
        critic_loss.backward(retain_graph=True) 
        actor_loss.backward()

        # Update both networks
        critic_optim.step()
        actor_optim.step()

        # Update priorities
        for i, episode in enumerate(batch):
            ep_loss = critic_losses[i].detach().item() * T
            ep_priority = actor_gradients[i] * config['lambda0'] + ep_loss
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
            print(f"Epoch {epoch+1} | Critic Loss: {critic_loss.item():.6f} | "
                  f"Actor Loss: {actor_loss.item():.6f}")
            
            save_model(agent, filename=f"trained_irdpg_{epoch+1}.pth", checkpoint_folder=run_folder[1])

    os.makedirs(run_folder[-1], exist_ok=True)
    plt.figure()
    plt.plot(total_critic_losses, label="Critic Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(run_folder[-1]+"/critic_loss_plot.png")
    plt.close()

    plt.figure()
    plt.plot(total_actor_losses, label="Actor Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(run_folder[-1]+"/actor_loss_plot.png")
    plt.close()


config = {
    "epochs": 100,          # Total training epochs
    "batch_size": 32,       # Episodes per batch
    "gamma": 0.99,          # Discount factor
    "tau": 0.001,            # Target network update rate
    "lambda0": 0.6,         # Priority weight
    "lambda1": 0.8,         # Policy gradient weight
    "lambda2": 0.2,         # BC loss weight
    "actor_lr": 1e-4,       # Learning rates
    "critic_lr": 1e-3,
    "eps_demo": 0.1,        # Priority boost for demos
    "noise_std": 0.01,       # Exploration noise
    "min_demo_episodes": 10, # Min demo episodes to start
    "seq_len": 5          # Match window_size
}

# Define the input parameters:
model_folder = input("Folder name for the model: ")
if model_folder == "":
    model_folder = None
encoder = input("Encoder (defaults to gru): ")
if encoder == "":
    encoder = "gru"

# Define the model
env = POMDPTEnv(df, window_size=config["seq_len"])
agent = iRDPGAgent(obs_dim=env.observation_space.shape[0], encoder_type=encoder, device=device)
buffer = PERBuffer(max_episodes=15)

train(config, env, agent, buffer, model_folder=model_folder)

