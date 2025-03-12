import torch
import numpy as np

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
    def __init__(self, max_episodes=100, alpha=0.6, beta=0.4, device="cuda"):
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
    

def generate_demonstration_episodes(env, n_episodes=50):
    episodes = []

    valid_days = env.valid_days
    selected_days = np.random.choice(valid_days, size=n_episodes, replace=True)

    for day in selected_days:
        env.current_day = day
        env.day_indices = np.where(env.dates == day)[0]
        env.reset()

        episode = Episode()
        episode.is_demo = True
        done = False
        
        while not done:
            action = dt_policy(env)
            obs, reward, done, _ = env.step(action)
            
            episode.obs.append(obs)
            episode.actions.append(action)
            episode.rewards.append(reward)
            episode.expert_actions.append([None, None])
            episode.dones.append(done)
    
        episodes.append(episode)

    print(f"Generated {len(episodes)} demonstration episodes.")
    return episodes


def collect_episode(env, agent):
    env.reset()
    episode = Episode()
    h_actor = None
    prev_action = None
    done = False

    # This now uses day-based indices internally
    expert_action = intraday_greedy_actions(env)
    expert_action_one_hot = np.zeros((len(expert_action), 2), dtype=np.float32)
    for i, a in enumerate(expert_action):
        if a == 1:
            expert_action_one_hot[i] = [1.0, 0.0]
        elif a == -1:
            expert_action_one_hot[i] = [0.0, 1.0]
        else:
            raise ValueError(f"Invalid action: {a}")

    while not done:
        obs = env._next_observation()
        action, h_actor = agent.act(obs, prev_action, h_actor)

        next_obs, reward, done, _ = env.step(action)
        
        episode.obs.append(obs)
        episode.actions.append(action)
        episode.rewards.append(reward)
        episode.expert_actions.append(expert_action_one_hot[env.current_step - env.day_start - 1])
        episode.dones.append(done)

        obs = next_obs
        
    return episode