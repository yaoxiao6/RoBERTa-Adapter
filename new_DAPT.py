import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score
import numpy as np
from util import save_checkpoint, load_checkpoint, check_checkpoint

# Assuming CUDA is available and being used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="RoBERTa-DAPT",

    # track hyperparameters and run metadata
    config={
        "learning_rate": 2e-5,
        "dataset": "problems_and_types",
        "epochs": 3,
    }
)

class Config:
    model_name = "./domain_adapted_roberta"
    learning_rate = 2e-5
    batch_size = 16
    num_epochs = 3 # 为什么只有3个 checkpoint，但是我们的最后的结果却有 12500 iteration?
    num_classes = 7
    hidden_size = 768

tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
base_model = AutoModel.from_pretrained(Config.model_name, output_hidden_states=True).to(device)
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
test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)

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
    for epoch in range(Config.num_epochs):
        for text, labels in dataloader:
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
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        wandb.log({"Epoch": epoch+1, "loss": loss.item()})
        save_checkpoint(model, classifier, optimizer, epoch + 1, checkpoint_dir)

def evaluate(model, classifier, dataloader):
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

train(base_model, classifier, train_loader, optimizer)
evaluate(base_model, classifier, test_loader)

wandb.finish()
