import numpy as np
import copy

import torch
import torch.nn as nn

from tqdm import trange, tqdm

from scripts.Env import intraday_greedy_actions, dt_policy


# EPISODES 
class Episode:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.expert_actions = []  # 'prophetic' (intraday actions)
        self.dones = [] # 'done' flags
        self.is_demo = False # if the episode is a demonstration
        self.priority = 1.0 # priority for PER

class PERBuffer:
    def __init__(self, max_episodes=1000, alpha=0.6, beta=0.4, device="cuda"):
        self.max_episodes = max_episodes
        self.alpha = alpha # priority exponent
        self.beta = beta # importance sampling exponent
        self.device = device
        self.episodes = []
        self.priorities = [] # for PER
    
    def add_episode(self, episode, eps_d=0.1):
        # Remove the oldest episode if buffer is full
        if len(self.episodes) >= self.max_episodes:
            self.episodes.pop(0)
            self.priorities.pop(0)

        episode.obs = torch.tensor(np.array(episode.obs, dtype=np.float32), device=self.device)
        episode.actions = torch.tensor(np.array(episode.actions, dtype=np.float32), device=self.device)
        episode.rewards = torch.tensor(np.array(episode.rewards, dtype=np.float32), device=self.device)
        episode.expert_actions = torch.tensor(np.array(episode.expert_actions, dtype=np.float32), device=self.device)
        episode.dones = torch.tensor(np.array(episode.dones, dtype=bool), device=self.device)

        if episode.is_demo:
            episode.priority += eps_d
        
        self.episodes.append(episode)
        self.priorities.append(episode.priority ** self.alpha)

    def update_priorities(self, indices, new_priorities):
        # Eq (10)
        for i, p in zip(indices, new_priorities):
            self.priorities[i] = p ** self.alpha

    def sample(self, batch_size):
        probs = np.array(self.priorities) / np.sum(self.priorities)
        indices = np.random.choice(len(self.episodes), batch_size, p=probs)
        batch = [self.episodes[i] for i in indices]
        
        # Eq (10)
        weights = (len(self.episodes) * probs[indices]) ** (-self.beta)
        weights = weights / np.max(weights)

        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        return batch, indices, weights
    
    def __len__(self):
        return len(self.episodes)
    
# AGENT

class iRDPGAgent(nn.Module):
    def __init__(self, 
                 obs_dim, 
                 action_dim=2, 
                 hidden_dim=64, 
                 tau=0.001,
                 device="cuda"):
        super().__init__()

        self.actor_gru = nn.GRU(obs_dim, hidden_dim, batch_first=True).to(device)
        self.actor_fc = nn.Linear(hidden_dim, action_dim).to(device) # [P long, P short]

        self.critic_gru = nn.GRU(obs_dim + action_dim, hidden_dim, batch_first=True).to(device)
        self.critic_fc = nn.Linear(hidden_dim, 1).to(device)

        # Target networks
        self.target_actor = copy.deepcopy(self.actor_gru)
        self.target_actor_fc = copy.deepcopy(self.actor_fc)
        self.target_critic = copy.deepcopy(self.critic_gru)
        self.target_critic_fc = copy.deepcopy(self.critic_fc)

        # Initalize target networks
        self.target_actor.flatten_parameters()
        self.target_critic.flatten_parameters()

        self._update_target_networks(tau)

        self.device = device

    def forward(self, obs, h_actor=None, h_critic=None):
        obs = obs.to(self.device)
        h_actor = h_actor.to(self.device) if h_actor is not None else None
        h_critic = h_critic.to(self.device) if h_critic is not None else None
        
        # Actor
        z_actor, h_actor_next = self.actor_gru(obs, h_actor)
        action_probs = torch.softmax(self.actor_fc(z_actor), dim=-1)

        # Critic
        z_critic, h_critic_next = self.critic_gru(torch.concat([obs, action_probs.detach()], dim=-1), h_critic)
        q_value = self.critic_fc(z_critic)

        return action_probs, q_value, h_actor_next, h_critic_next
    
    def target_forward(self, obs, h_actor=None, h_critic=None):
        obs = obs.to(self.device).contiguous()
        h_actor = h_actor.to(self.device).contiguous() if h_actor is not None else None
        h_critic = h_critic.to(self.device).contiguous() if h_critic is not None else None

        # Target actor
        z_actor, h_actor_next = self.target_actor(obs, h_actor)
        target_action_probs = torch.softmax(self.target_actor_fc(z_actor), dim=-1)

        # Target critic
        z_critic, h_critic_next = self.target_critic(torch.concat([obs, target_action_probs.detach()], dim=-1).contiguous(), h_critic)
        target_q_value = self.target_critic_fc(z_critic)

        return target_action_probs, target_q_value, h_actor_next, h_critic_next
    
    
    def critic_forward(self, obs, action, h_critic=None):
        obs = obs.to(self.device)
        action = action.to(self.device)
        h_critic = h_critic.to(self.device) if h_critic is not None else None

        z_critic, h_critic_next = self.critic_gru(torch.concat([obs, action], dim=-1), h_critic)
        q_value = self.critic_fc(z_critic)

        return q_value, h_critic_next
        
    
    def act(self, obs, h_actor=None, add_noise=True):

        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        h_actor = h_actor.to(self.device) if h_actor is not None else None

        with torch.no_grad():
            action_probs, _, h_actor_next, _ = self.forward(obs, h_actor)
            
        action = action_probs.squeeze(0).cpu().numpy()

        if add_noise:
            noise = np.random.normal(0, 0.1, size=action.shape)
            action = np.clip(action + noise, 0, 1)

            
        return torch.tensor(action, dtype=torch.float32), h_actor_next
    
    def _update_target_networks(self, tau):
        # Polyak averaging
        for target_param, param in zip(self.target_actor.parameters(), self.actor_gru.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_actor_fc.parameters(), self.actor_fc.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic_gru.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic_fc.parameters(), self.critic_fc.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

def generate_demonstration_episodes(env, n_episodes=50):
    episodes = []
    for _ in trange(n_episodes, desc='Generating Demonstrations'):
        env.reset()
        episode = Episode()
        episode.is_demo = True
        done = False

        expert_action = intraday_greedy_actions(env)

        expert_action_one_hot = np.zeros((len(expert_action), 2), dtype=np.float32)
        for i, a in enumerate(expert_action):
            if a == 0:
                expert_action_one_hot[i] = [1.0, 0.0]
            elif a == 1:
                expert_action_one_hot[i] = [0.0, 1.0]
            else:
                raise ValueError(f"Invalid action: {a}")
        
        
        while not done:
            action = dt_policy(env)
            obs, reward, done, _ = env.step(action)
            
            episode.obs.append(obs)
            episode.actions.append(action)
            episode.rewards.append(reward)
            episode.expert_actions.append(expert_action_one_hot[env.current_step-1])
            episode.dones.append(done)
        
        episodes.append(episode)
    return episodes

def collect_episode(env, agent, add_noise):
    env.reset()
    episode = Episode()
    h_actor = None
    done = False

    expert_action = intraday_greedy_actions(env)
    expert_action_one_hot = np.zeros((len(expert_action), 2), dtype=np.float32)
    for i, a in enumerate(expert_action):
        if a == 0:
            expert_action_one_hot[i] = [1.0, 0.0]
        elif a == 1:
            expert_action_one_hot[i] = [0.0, 1.0]
        else:
            raise ValueError(f"Invalid action: {a}")

    while not done:
        obs = env._next_observation()
        action, h_actor = agent.act(obs, h_actor, add_noise=add_noise)
        
        next_obs, reward, done, _ = env.step(action)
        
        episode.obs.append(obs)
        episode.actions.append(action)
        episode.rewards.append(reward)
        episode.expert_actions.append(expert_action_one_hot[env.current_step-1])
        episode.dones.append(done)

        obs = next_obs
        
    return episode
            
