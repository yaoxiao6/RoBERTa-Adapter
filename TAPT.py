# TAPT.py

from datasets import load_dataset
from transformers import RobertaTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import RobertaForMaskedLM, Trainer, TrainingArguments

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

training_args = TrainingArguments(
    output_dir="./tapt_pretraining",
    overwrite_output_dir=True,
    num_train_epochs = 100,
    per_device_train_batch_size=16,
    save_steps=10000,
    prediction_loss_only=True,
    gradient_accumulation_steps=128,
    learning_rate=0.0001
)


# Initialize Trainer
TATP_trainer = Trainer(
    model=cur_model,
    args=training_args,
    train_dataset=tokenized_tapt_train_dataset['train'],
    data_collator=data_collator,
)

# Start pretraining
TATP_trainer.train()

# Save the domain-adapted model and tokenizer
cur_model.save_pretrained('./TATP_roberta')
tokenizer.save_pretrained('./TATP_roberta')
