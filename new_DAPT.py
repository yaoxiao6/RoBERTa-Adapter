import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score
import numpy as np
from util import save_checkpoint, load_checkpoint, check_checkpoint
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

# import wandb
# wandb.login()
# # start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="my-awesome-project",

#     # track hyperparameters and run metadata
#     config={
#         "learning_rate": 2e-5,
#         "architecture": "Roberta-DAPT",
#         "dataset": "CIFAR-100",
#         "epochs": 3,
#         "batch_size": 16,
#         "num_epochs": 3,
#         "num_classes": 7,
#         "hidden_size": 768
#     }
# )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_dir = "./checkpoints"

class Config:
    model_name = "./domain_adapted_roberta"
    fallback_model_name = "roberta-base"
    learning_rate = 2e-5 
    batch_size = 16
    num_epochs = 3
    num_classes = 7
    hidden_size = 768

# try:
#     tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
#     base_model = AutoModel.from_pretrained(Config.model_name, output_hidden_states=True).to(device)
# except OSError:
# print(f"Warning: {Config.model_name} not found or missing required files. Using {Config.fallback_model_name} as a fallback.")
tokenizer = AutoTokenizer.from_pretrained(Config.fallback_model_name)
base_model = AutoModel.from_pretrained(Config.fallback_model_name, output_hidden_states=True).to(device)

for param in base_model.parameters():
    param.requires_grad = False
tokenizer = AutoTokenizer.from_pretrained(Config.fallback_model_name)
base_model = AutoModel.from_pretrained(Config.fallback_model_name, output_hidden_states=True).to(device)
for param in base_model.parameters():
    param.requires_grad = False

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

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
print("============== aaaa =============")
print(test_dataset.get(0))
print("Content of test_dataset:")
for text, label in test_dataset:
    print(f"Text: {text}")
    print(f"Label: {label}")
    print("---")
    
test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)
print("Content of test_loader:")
print("First 2 batches of test_loader:")
for i, batch in enumerate(test_loader):
    if i >= 2:
        break
    texts, labels = batch
    print(f"Batch size: {len(texts)}")
    print("Texts:")
    for text in texts:
        print(text)
    print("Labels:", labels)
    print("---")

class Classifier(torch.nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.linear(x)

classifier = Classifier(hidden_size=Config.hidden_size, num_classes=Config.num_classes).to(device)
optimizer = optim.Adam(classifier.parameters(), lr=Config.learning_rate)

def train(model, classifier, dataloader, optimizer):
    model.eval()
    classifier.train()
    start_epoch = 0
    checkpoint_path = check_checkpoint(checkpoint_dir)
    if checkpoint_path:
        print(f"Loading checkpoint from: {checkpoint_path}")
        model, classifier, optimizer, start_epoch = load_checkpoint(model, classifier, optimizer, checkpoint_path)
    
    num_batches = len(dataloader)
    
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
            
            progress_bar.set_postfix({"Loss": loss.item()})
            # log metrics to wandb
            wandb.log({"loss": loss.item()})
        
        save_checkpoint(model, classifier, optimizer, epoch + 1, checkpoint_dir)

def evaluate(model, classifier, dataloader, checkpoint_dir):
    checkpoint_path = check_checkpoint(checkpoint_dir)
    if checkpoint_path:
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        print("Model and classifier loaded from checkpoint.")
    else:
        print("No checkpoint found. Using the current model and classifier.")
    
    model.eval()
    classifier.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for text, labels in dataloader:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
            labels = torch.tensor(labels).to(device)
            outputs = model(**inputs)
            cls_embeddings = outputs.hidden_states[-1][:, 0, :]
            logits = classifier(cls_embeddings)
            predictions = torch.argmax(logits, dim=1).to(device)
            all_predictions.extend(predictions.numpy()).to(device)
            all_labels.extend(labels.numpy())
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    print(f"Test F1 Score: {f1}")
    wandb.log({"test_f1_score": f1_score})

# train(base_model, classifier, train_loader, optimizer)
evaluate(base_model, classifier, test_loader, checkpoint_dir)
