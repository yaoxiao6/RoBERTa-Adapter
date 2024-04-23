/* util.py */

import os
import torch

def save_checkpoint(model, classifier, optimizer, scheduler, epoch, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)  # Create the checkpoint directory if it doesn't exist
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'classifier_state_dict': classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch
    }
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at: {checkpoint_path}")

def load_checkpoint(model, classifier, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    print("checkpoint = ", checkpoint['optimizer_state_dict'].keys())
    model.load_state_dict(checkpoint['model_state_dict'])
    print("model = ", model)
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    print("classifier = ", classifier)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("optimizer = ", optimizer)
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print("scheduler = ", scheduler)
    epoch = checkpoint['epoch']
    print("epoch = ", epoch)
    return model, classifier, optimizer, scheduler, epoch

def check_checkpoint(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_')]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return os.path.join(checkpoint_dir, latest_checkpoint)
    return None