# TAPT.py

import torch 
from datasets import load_dataset
from transformers import RobertaTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import RobertaForMaskedLM, Trainer, TrainingArguments
from transformers import RobertaTokenizer
from transformers.integrations import WandbCallback  # Import the WandbCallback

import wandb
wandb.login()
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",

    # track hyperparameters and run metadata
    config={
        "Name": "TAPT.py",
        "dataset": "problems_and_types",
        "output_dir": "./tapt_pretraining",
        "prediction_loss_only": True,
        "num_train_epochs": 5,
        "per_device_train_batch_size": 16,
        "gradient_accumulation_steps": 128,
        "weight_decay": 0.01,
        "save_steps": 200,  # Save checkpoint every 200 steps
        "logging_steps": 1,  # Log loss and learning rate every step
        "optimizer" : "AdamW(cur_model.parameters(), lr=lr=0.01, eps=1e-8)",
        "scheduler" : "StepLR(optimizer, step_size=200, gamma=0.5)",        
        "data_collator": "DataCollatorForLanguageModeling(tokenizer=tapt_tokenizer,mlm=True,mlm_probability=0.15)",
    }
)
tapt_train_dataset = load_dataset('csv', data_files={'train': 'Dataset/problems_and_types_train.csv'})

# Tokenization of the dataset
tapt_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
def tokenize_function(examples):
    return tapt_tokenizer(examples['problem+answer'], padding="max_length", truncation=True, max_length=100)

tokenized_tapt_train_dataset = tapt_train_dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tapt_tokenizer,
    mlm=True,
    mlm_probability=0.15  # 15% of the tokens will be masked
)

# Load your fine-tuned model
cur_model = RobertaForMaskedLM.from_pretrained('roberta-base')

# Define the optimizer and scheduler
optimizer = torch.optim.AdamW(cur_model.parameters(), lr=0.01, eps=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

training_args = TrainingArguments (
    output_dir="./tapt_pretraining",
    prediction_loss_only=True,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=128,
    weight_decay=0.01,
    save_steps=200,  # Save checkpoint every 200 steps
    logging_steps=1,  # Log loss and learning rate every step
    report_to='wandb',  # Enable logging to wandb
)

TATP_trainer = Trainer(
    model=cur_model,
    args=training_args,
    train_dataset=tokenized_tapt_train_dataset['train'],
    data_collator=data_collator,
    optimizers=(optimizer, scheduler),  # Pass the optimizer and scheduler
    callbacks=[WandbCallback()],  # Add the WandbCallback to the callbacks list
)

# Start pretraining
TATP_trainer.train()

# Save the domain-adapted model and tokenizer
cur_model.save_pretrained('./TATP_roberta')
tapt_tokenizer.save_pretrained('./TATP_roberta')
