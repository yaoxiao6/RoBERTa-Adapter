# DAPT.py

import torch 
from datasets import load_dataset
from transformers import RobertaTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import RobertaForMaskedLM, Trainer, TrainingArguments
from transformers import RobertaTokenizer
from transformers.integrations import WandbCallback

import wandb
wandb.login()
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",

    # track hyperparameters and run metadata
    config={
        "Name": "DAPT.py",
        "dataset": "Dataset/30000_formula.csv",
        "output_dir": "./DAPT_pretraining",
        "prediction_loss_only": True,
        "num_train_epochs": 5,
        "per_device_train_batch_size": 16,
        "gradient_accumulation_steps": 128,
        "weight_decay": 0.01,
        "save_steps": 200,  # Save checkpoint every 200 steps
        "logging_steps": 1,  # Log loss and learning rate every step
        "optimizer" : "AdamW(cur_model.parameters(), lr=lr=0.01, eps=1e-8)",
        "scheduler" : "StepLR(optimizer, step_size=200, gamma=0.5)",        
        "data_collator": "DataCollatorForLanguageModeling(tokenizer=dapt_tokenizer,mlm=True,mlm_probability=0.15)",
    }
)
dapt_train_dataset = load_dataset('csv', data_files={'train': 'Dataset/30000_formula.csv'})

# Tokenization of the dataset
dapt_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
def tokenize_function(examples):
    return dapt_tokenizer(examples['formula'], padding="max_length", truncation=True, max_length=50)

tokenized_dapt_train_dataset = dapt_train_dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=dapt_tokenizer,
    mlm=True,
    mlm_probability=0.15  # 15% of the tokens will be masked
)

# Load your fine-tuned model
cur_model = RobertaForMaskedLM.from_pretrained('roberta-base')

# Define the optimizer and scheduler
optimizer = torch.optim.AdamW(cur_model.parameters(), lr=0.01, eps=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

training_args = TrainingArguments (
    output_dir="./dapt_pretraining",
    prediction_loss_only=True,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=128,
    weight_decay=0.01,
    save_steps=200,  # Save checkpoint every 200 steps
    logging_steps=1,  # Log loss and learning rate every step
    report_to='wandb',  # Enable logging to wandb
)

DAPT_trainer = Trainer(
    model=cur_model,
    args=training_args,
    train_dataset=tokenized_dapt_train_dataset['train'],
    data_collator=data_collator,
    optimizers=(optimizer, scheduler),  # Pass the optimizer and scheduler
    callbacks=[WandbCallback()],  # Add the WandbCallback to the callbacks list
)

# Start pretraining
DAPT_trainer.train()

# Save the domain-adapted model and tokenizer
cur_model.save_pretrained('./DAPT_roberta')
dapt_tokenizer.save_pretrained('./DAPT_roberta')
