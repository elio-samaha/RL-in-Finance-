import os

import torch

def save_model(agent, filename="trained_irdpg.pth", run_folder=""):
    path_dir = run_folder
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
    torch.save(checkpoint, path_dir+'/'+filename)
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