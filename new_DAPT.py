### DAPT.py ###

import os
from tqdm import tqdm
import pandas as pd
import numpy as np

from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import f1_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from util import save_checkpoint, load_checkpoint, check_checkpoint

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Dataset

import wandb
wandb.login()
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",

    # track hyperparameters and run metadata
    config={
        "architecture": "Roberta-DAPT",
        "dataset": "CIFAR-100",
        "model_name": "./domain_adapted_roberta",
        "fallback_model_name": "roberta-base",
        "initial_learning_rate": 0.5,
        "batch_size": 16,
        "num_epochs": 3,
        "num_classes": 7,
        "hidden_size": 768,
        "scheduler_step_size": 200,
        "scheduler_gamma": 0.6
    }
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_dir = "./checkpoints/checkpoints-DAPT-run3"

class Config:
    model_name = "./domain_adapted_roberta"
    fallback_model_name = "roberta-base"
    initial_learning_rate = 0.5
    batch_size = 16
    num_epochs = 3
    num_classes = 7
    hidden_size = 768
    scheduler_step_size = 200
    scheduler_gamma = 0.6

# try:
#     tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
#     base_model = AutoModel.from_pretrained(Config.model_name, output_hidden_states=True).to(device)
# except OSError:
# print(f"Warning: {Config.model_name} not found or missing required files. Using {Config.fallback_model_name} as a fallback.")
tokenizer = AutoTokenizer.from_pretrained(Config.fallback_model_name)
base_model = AutoModel.from_pretrained(Config.fallback_model_name, output_hidden_states=True).to(device)
for param in base_model.parameters():
    param.requires_grad = False

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts # text is a pandas series
        self.labels = labels # labels is a list

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # print(
        #     "self.texts type = ", type(self.texts),
        #     "self.labels type = ", type(self.labels),
        # )
        return self.texts.iloc[idx], self.labels[idx]

# Example data
# texts = ["example text 1", "example text 2", "example text 3"]
# labels = [0, 1, 2]
csv_file_path1 = 'Dataset/problems_and_types_train.csv'
csv_file_path2 = 'Dataset/problems_and_types_test.csv'
df1 = pd.read_csv(csv_file_path1)
df2 = pd.read_csv(csv_file_path2)
train_texts, train_labels = df1['problem+answer'], df1['type']
test_texts, val_texts, test_labels, val_labels = train_test_split(df2['problem+answer'], df2['type'], test_size=0.5, random_state=42)
label_dict = {label: i for i, label in enumerate(df1['type'].unique())}
train_labels = [label_dict[label] for label in train_labels]
val_labels = [label_dict[label] for label in val_labels]
test_labels = [label_dict[label] for label in test_labels]

train_dataset = TextDataset(train_texts, train_labels)
train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)

# Example test data
test_dataset = TextDataset(test_texts, test_labels)
# print("============== Yao print debug =============")
# print(test_dataset)
# print(test_dataset[0])
# print("Content of test_dataset:")
# for text, label in test_dataset:
#     print(f"Text: {text}")
#     print(f"Label: {label}")
#     print("---")

test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)
# print("Content of test_loader:")
# print("First 2 batches of test_loader:")
# for i, batch in enumerate(test_loader):
#     if i >= 2:
#         break
#     texts, labels = batch
#     print(f"Batch size: {len(texts)}")
#     print("Texts:")
#     for text in texts:
#         print(text)
#     print("Labels:", labels)
#     print("---")

class Classifier(torch.nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.linear(x)

classifier = Classifier(hidden_size=Config.hidden_size, num_classes=Config.num_classes).to(device)
optimizer = optim.Adam(classifier.parameters(), lr=Config.initial_learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.scheduler_step_size, gamma=Config.scheduler_gamma)

def train(model, classifier, dataloader, optimizer, scheduler):
    model.eval()
    classifier.train()
    start_epoch = 0
    checkpoint_path = check_checkpoint(checkpoint_dir)
    if checkpoint_path:
        print(f"Loading checkpoint from: {checkpoint_path}")
        model, classifier, optimizer, scheduler, start_epoch = load_checkpoint(model, classifier, optimizer, scheduler, checkpoint_path)
    
    num_batches = len(dataloader)
    global_step = start_epoch * num_batches
    
    for epoch in range(start_epoch, start_epoch + Config.num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{start_epoch + Config.num_epochs}", unit="batch")
        
        for text, labels in progress_bar:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
            labels = torch.tensor(labels).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                cls_embeddings = outputs.hidden_states[-1][:, 0, :]
            logits = classifier(cls_embeddings)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({"Loss": loss.item(), "LR": scheduler.get_last_lr()[0]})
            wandb.log({"loss": loss.item(), "learning_rate": scheduler.get_last_lr()[0]}, step=global_step)
            global_step += 1
        
        save_checkpoint(model, classifier, optimizer, scheduler, epoch + 1, checkpoint_dir)


def predict(model, classifier, dataloader):
    model.eval()
    classifier.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Prediction", unit="batch")
        for text, labels in progress_bar:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
            labels = torch.tensor(labels).to(device)
            outputs = model(**inputs)
            cls_embeddings = outputs.hidden_states[-1][:, 0, :]
            logits = classifier(cls_embeddings)
            predictions = torch.argmax(logits, dim=1).to(device)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({"all_predictions": len(all_predictions)})
    
    return all_labels, all_predictions

def evaluate(all_labels, all_predictions):
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    print(f"Test F1 Score: {f1}")
    recall = recall_score(all_labels, all_predictions, average='weighted')
    print(f"Test Recall: {recall}")
    cm = confusion_matrix(all_labels, all_predictions)
    print(f"Confusion Matrix:\n{cm}")
    
    wandb.log({
        "test_f1_score": f1,
        "test_recall": recall,
        "test_confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=all_labels, preds=all_predictions, class_names=list(label_dict.keys())),
    })
    return f1, recall, cm


# reload the model and classifier
checkpoint_path = check_checkpoint(checkpoint_dir)
if checkpoint_path:
    print(f"Loading checkpoint from: {checkpoint_path}")
    model, classifier, _, _, _ = load_checkpoint(base_model, classifier, optimizer, scheduler, checkpoint_path)
    print("Model and classifier loaded from checkpoint.")
else:
    raise ValueError("No checkpoint found. Please train the model first.")
# train(base_model, classifier, train_loader, optimizer, scheduler)


# Reload predictions
if os.path.exists('all_labels.pt') and os.path.exists('all_predictions.pt'):
    print("Loading predictions from files...")
    all_labels = torch.load('all_labels.pt')
    all_predictions = torch.load('all_predictions.pt')
else:
    print("Running predictions...")
    all_labels, all_predictions = predict(model, classifier, test_loader)
    torch.save(all_labels, 'all_labels.pt')
    torch.save(all_predictions, 'all_predictions.pt')

# evaluate
evaluate(all_labels, all_predictions)
