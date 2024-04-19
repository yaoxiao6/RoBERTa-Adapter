import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""# DATP

"""


# # Example: load your domain-specific dataset
# # This is an example, replace 'text_file' and 'file_path' with your domain-specific data paths

# formula_dataset = load_dataset('csv', data_files={'train': 'Dataset/half_formula.csv'})

# # Tokenization of the dataset
# from transformers import RobertaTokenizer

# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# def tokenize_function(examples):
#     return tokenizer(examples['formula'], padding="max_length", truncation=True, max_length=512)

# tokenized_datasets = formula_dataset.map(tokenize_function, batched=True)

formula_dataset_30000 = load_dataset('csv', data_files={'train': 'Dataset/30000_formula.csv'})

# Tokenization of the dataset
from transformers import RobertaTokenizer

tokenizer_30000 = RobertaTokenizer.from_pretrained('roberta-base')
def tokenize_function(examples):
    return tokenizer_30000(examples['formula'], padding="max_length", truncation=True, max_length=50)

tokenized_datasets_30000 = formula_dataset_30000.map(tokenize_function, batched=True)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer_30000,
    mlm=True,
    mlm_probability=0.15  # 15% of the tokens will be masked
)

# from transformers import DataCollatorWithPadding

# data_collator = DataCollatorWithPadding(tokenizer=tokenizer_30000, pad_to_multiple_of=8)


from transformers import RobertaForMaskedLM, Trainer, TrainingArguments

# Load your fine-tuned model
DATPmodel = RobertaForMaskedLM.from_pretrained('roberta-base')


training_args = TrainingArguments(
    output_dir="./domain_adaptive_pretraining",
    overwrite_output_dir=True,
    max_steps=12500,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    # gradient_accumulation_steps=8,
    save_steps=10000,
    prediction_loss_only=True,
    fp16=False,  # Enable mixed precision
)


# Initialize Trainer
DATP_trainer = Trainer(
    model=DATPmodel,
    args=training_args,
    train_dataset=tokenized_datasets_30000['train'],
    data_collator=data_collator,
)

# Start pretraining
DATP_trainer.train()

# Save the domain-adapted model and tokenizer
DATPmodel.save_pretrained('./domain_adapted_roberta')
tokenizer.save_pretrained('./domain_adapted_roberta') # NameError: name 'tokenizer' is not defined

# Load the domain-adapted RoBERTa model for sequence classification
DATP_Class_model = RobertaForSequenceClassification.from_pretrained(DATPmodel, num_labels=len(label_dict))
tokenizer = RobertaTokenizer.from_pretrained(DATPmodel)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results_type_classification',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    adam_epsilon = 1e-6,
    warmup_ratio = 0.06,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Initialize Trainer
DATP_Class_trainer = Trainer(
    model=DATP_Class_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
DATP_Class_trainer.train()

DATP_Class_model.save_pretrained('./DATP_Class_model')
tokenizer.save_pretrained('./DATP_Class_model')

predictions = finetune_trainer.predict(test_dataset)
pred_labels = np.argmax(predictions.predictions, axis=1)
f1 = f1_score(val_labels, pred_labels, average='weighted')
print(f'DATP F1 Score: {f1}')

