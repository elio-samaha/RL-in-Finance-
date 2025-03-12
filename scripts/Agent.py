import copy
import numpy as np

import torch
import torch.nn as nn
    
# AGENT
class iRDPGAgent(nn.Module):
    def __init__(self, 
                 obs_dim, 
                 action_dim=2, 
                 hidden_dim=64, 
                 tau=0.001,
                 encoder_type="gru",
                 device="cuda"):
        super().__init__()

        self.encoder_type = encoder_type

        # GRU encoder
        if self.encoder_type=='gru':
            self.encoder = nn.GRU(action_dim + obs_dim, hidden_dim, batch_first=True).to(device)
            
        elif self.encoder_type=='transformer':
            # Projection layer
            self.input_projection = nn.Linear(obs_dim + action_dim, hidden_dim).to(device)
            # Positional encoding
            self.positional_encoding = PositionalEncoding(hidden_dim).to(device)
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, 
                nhead=4, 
                dim_feedforward=hidden_dim,
                batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2).to(device)

        elif self.encoder_type=='lstm':
            self.encoder = nn.LSTM(action_dim + obs_dim, hidden_dim, batch_first=True).to(device)

        elif self.encoder_type=='mlp':
            self.encoder = nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ).to(device)
            
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        # Actor
        self.actor_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True).to(device)
        self.actor_fc = nn.Linear(hidden_dim, action_dim).to(device) # [P long, P short]

        # Critic
        self.critic_gru = nn.GRU(hidden_dim + action_dim, hidden_dim, batch_first=True).to(device)
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
        self.action_dim = action_dim
        self.obs_dim = obs_dim

    def encode(self, x, h_t):
        x = x.unsqueeze(0).to(self.device)
        h_t = h_t.unsqueeze(0).to(self.device) if h_t is not None else None

        if self.encoder_type == 'gru':
            _, h = self.encoder(x)
            return h
        
        elif self.encoder_type == 'transformer':

            x = self.input_projection(x)

            if h_t is None:
                memory = x
            else:
                memory = torch.concat([h_t, x], dim=1)

            x = self.positional_encoding(memory)
            x = x.transpose(0, 1)  # [batch_size, hidden_dim, seq_len]
            x = self.encoder(x)
            x = x.transpose(0, 1)  # [seq_len, batch_size, hidden_dim]

            return x[:, -1, :]  # Return the last hidden state
        
        elif self.encoder_type == 'lstm':
            if h_t is None:
                _, (h, c) = self.encoder(x, None)
            else:
                h_t = h_t.squeeze(0)
                c_t = torch.zeros_like(h_t).to(self.device)
                _, (h, c) = self.encoder(x, (h_t, c_t))
            return h


    def actor_forward(self, obs, h_actor=None):
        obs = obs.to(self.device)
        h_actor = h_actor.to(self.device) if h_actor is not None else None
        
        z_actor, h_actor_next = self.actor_gru(obs, h_actor)
        action_probs = torch.softmax(self.actor_fc(z_actor), dim=-1)

        return action_probs, h_actor_next
    
    def critic_forward(self, h_t, action, h_critic=None):
        h_t = h_t.to(self.device)
        action = action.to(self.device)
        h_critic = h_critic.to(self.device) if h_critic is not None else None

        z_critic, h_critic_next = self.critic_gru(torch.concat([h_t, action], dim=-1), h_critic)
        q_value = self.critic_fc(z_critic)

        return q_value, h_critic_next
    
    
    def target_forward(self, h_t, h_actor=None, h_critic=None):
        h_t = h_t.to(self.device).contiguous()
        h_actor = h_actor.to(self.device).contiguous() if h_actor is not None else None
        h_critic = h_critic.to(self.device).contiguous() if h_critic is not None else None

        # Target actor
        z_actor, h_actor_next = self.target_actor(h_t, h_actor)
        target_action_probs = torch.softmax(self.target_actor_fc(z_actor), dim=-1)

        # Target critic
        z_critic, h_critic_next = self.target_critic(torch.concat([h_t, target_action_probs.detach()], dim=-1).contiguous(), h_critic)
        target_q_value = self.target_critic_fc(z_critic)

        return target_action_probs, target_q_value, h_actor_next, h_critic_next
        
    
    def act(self, obs, prev_action=None, h_t=None, h_actor=None):

        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        h_actor = h_actor.to(self.device) if h_actor is not None else None

        if prev_action is None:
            prev_action = torch.zeros((1, self.action_dim), dtype=torch.float32, device=self.device)
        else:
            prev_action = torch.tensor(prev_action, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            h_t = self.encode(torch.cat([obs, prev_action], dim=-1), h_t)
            action_probs, h_actor_next = self.actor_forward(h_t, h_actor)
            action_probs = action_probs.squeeze(0).squeeze(0).cpu().numpy()
        return action_probs, h_actor_next
    
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

            
class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, dropout=0.1, max_len=5000, device="cuda"):
        super(PositionalEncoding, self).__init__()
        self.model_dim = model_dim
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)
        
        self.pe = torch.zeros(max_len, model_dim, device=device)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        self.pe[:, 0::2] = torch.sin(np.pi * position / (max_len - 1))
        self.pe[:, 1::2] = torch.cos(np.pi/2 + np.pi * position / (max_len - 1))
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)     
