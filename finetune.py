import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, confusion_matrix, roc_auc_score
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import torch.nn.functional as F
import os


# import wandb
# wandb.login()
# # start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="my-awesome-project",

#     # track hyperparameters and run metadata
#     config={
#         "Name": "RoBERTa-Adapter finetune",
#         "dataset": "problems_and_types",
#     }
# )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
csv_file_path1 = './Dataset/problems_and_types_train.csv'
csv_file_path2 = './Dataset/problems_and_types_test.csv'

# Read the CSV files into DataFrames
df1 = pd.read_csv(csv_file_path1)
df2 = pd.read_csv(csv_file_path2)

# Train-test split (assuming you don't have separate training and validation sets)
train_texts, train_labels = df1['problem+answer'], df1['type']
test_texts, val_texts, test_labels, val_labels = train_test_split(df2['problem+answer'], df2['type'], test_size=0.5, random_state=42)

# Unique classes
label_dict = {label: i for i, label in enumerate(df1['type'].unique())}
train_labels = [label_dict[label] for label in train_labels]
val_labels = [label_dict[label] for label in val_labels]
test_labels = [label_dict[label] for label in test_labels]

val_texts.index = range(len(val_texts))
test_texts.index = range(len(test_texts))

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_token_len=100):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

train_dataset = TextDataset(train_texts, train_labels, tokenizer)
val_dataset = TextDataset(val_texts, val_labels, tokenizer)
test_dataset = TextDataset(test_texts, test_labels, tokenizer)

fintune_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(label_dict))

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    adam_epsilon = 1e-6,
    warmup_ratio = 0.06,
    weight_decay=0.01,
    logging_dir='./logs',
)

finetune_trainer = Trainer(
    model=fintune_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train and Save the model and the tokenizer
# finetune_trainer.train()
# fintune_model.save_pretrained('./finetuned_roberta')
# tokenizer.save_pretrained('./finetuned_roberta')

# Reload the saved model
def reload_model(model_path):
    reloaded_model = RobertaForSequenceClassification.from_pretrained(model_path)
    return reloaded_model

def predict(model, dataset):
    predictions = model.predict(dataset)
    # torch.save(predictions, 'predictions.pt')
    return predictions

def evaluate(predictions, test_labels):
    # print("============= Debugging =============== \n", predictions.type)
    pred_probs = F.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
    pred_labels = np.argmax(pred_probs, axis=1)
    
    f1 = f1_score(test_labels, pred_labels, average='weighted')
    print(f'Fine-tune F1 Score: {f1}') # Fine_tune F1 Score: 0.02256115107913669
    recall = recall_score(test_labels, pred_labels, average='weighted')
    print(f'Fine-tune Recall: {recall}') # Fine-tune Recall: 0.112
    cm = confusion_matrix(test_labels, pred_labels)
    #     Confusion Matrix:
    # [[  0   0   0   0   0   0 574]
    #  [  0   0   0   0   0   0 233]
    #  [  0   0   0   0   0   0 244]
    #  [  0   0   0   0   0   0 440]
    #  [  0   0   0   0   0   0 291]
    #  [  0   0   0   0   0   0 438]
    #  [  0   0   0   0   0   0 280]]
    print(f'Confusion Matrix:\n{cm}')
    # auc_roc = roc_auc_score(test_labels, predictions.predictions, multi_class='ovr')
    # print(f'AUC-ROC: {auc_roc}')
    
    # wandb.log({
    #     "test_f1_score": f1,
    #     "test_recall": recall,
    #     "test_confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=test_labels, preds=pred_labels, class_names=label_dict.keys()),
    #    #"test_auc_roc": auc_roc
    # })
    return f1, recall, cm    
    # return f1, recall, cm, auc_roc

reloaded_model = reload_model('./finetuned_roberta')
finetune_trainer.model = reloaded_model
if finetune_trainer.model is not None:
    print("Model is reloaded successfully")
    # Check if the predictions checkpoint exists
    if os.path.exists('predictions.pt'):
        print("Loading predictions from checkpoint...")
        predictions = torch.load('predictions.pt')
    else:
        print("Running predictions...")
        predictions = predict(finetune_trainer, test_dataset)
        torch.save(predictions, 'predictions.pt')

    evaluate(predictions, test_labels)