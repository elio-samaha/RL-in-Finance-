import os
import torch

def save_model(agent, filename="trained_irdpg.pth", checkpoint_folder=""):
    os.makedirs(checkpoint_folder, exist_ok=True)
    
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
    
    save_path = os.path.join(checkpoint_folder, filename)
    torch.save(checkpoint, save_path)
    print(f"\nModel saved as {save_path}\n")

def get_model_folder(base_dir="models/", folder_name=None):
    os.makedirs(base_dir, exist_ok=True)
    if folder_name is None:
        existing_runs = sorted(
            [d for d in os.listdir(base_dir) if d.isdigit()],
            key=lambda x: int(x)
        )
        next_run_number = 1 if not existing_runs else int(existing_runs[-1]) + 1
        folder_name = f"{next_run_number:02d}"

    run_folder = os.path.join(base_dir, folder_name)
    os.makedirs(run_folder, exist_ok=True)
    
    checkpoint_folder = os.path.join(run_folder, "checkpoint")
    os.makedirs(checkpoint_folder, exist_ok=True)
    
    images_folder = os.path.join(run_folder, "images")
    os.makedirs(images_folder, exist_ok=True)
    
    return run_folder, checkpoint_folder, images_folder