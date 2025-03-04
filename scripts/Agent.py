import numpy as np
import copy

import torch
import torch.nn as nn
    
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

    def forward(self, obs, h_actor=None):
        obs = obs.to(self.device)
        h_actor = h_actor.to(self.device) if h_actor is not None else None
        
        # Actor
        z_actor, h_actor_next = self.actor_gru(obs, h_actor)
        action_probs = self.actor_fc(z_actor)

        return action_probs, h_actor_next
    
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
            action_probs, h_actor_next = self.forward(obs, h_actor)
            
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
            
            
