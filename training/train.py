import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

import logging
import pandas as pd
import torch
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    TrainingArguments,
    Trainer,
    HfArgumentParser
)
from torch.utils.data import Dataset

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Verify CUDA availability
if not torch.cuda.is_available():
    logging.warning("CUDA is not available. Training will use CPU.")

# Logging script start
logging.info("----- Script Started -----")

# Load datasets
try:
    train_df = pd.read_csv("train.csv")
    eval_df = pd.read_csv("val.csv")
    logging.info("---- Data Loaded ----")
except FileNotFoundError as e:
    logging.error(f"Data file not found: {e}")
    exit(1)

# Training arguments
args_dict = {
    "output_dir": "modelname-ocr-correction-1024",
    "overwrite_output_dir": True,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-4,
    "warmup_steps": 250,
    "logging_steps": 100,
    "evaluation_strategy": "steps",
    "eval_steps": 250,
    "num_train_epochs": 4,
    "do_train": True,
    "do_eval": True,
    "fp16": torch.cuda.is_available(),
    "max_steps": 100000,
    "save_total_limit": 2  # Limit the number of saved checkpoints
}
parser = HfArgumentParser((TrainingArguments,))
training_args = parser.parse_dict(args_dict)[0]

# Load tokenizer and model
try:
    tokenizer = MBart50TokenizerFast.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt",
        cache_dir="cache",
        max_length=1024
    )
    model = MBartForConditionalGeneration.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt",
        cache_dir="cache"
    )
    logging.info("----- Model and Tokenizer Loaded -----")
except Exception as e:
    logging.error(f"Error loading model or tokenizer: {e}")
    exit(1)

# Update model and tokenizer configuration
tokenizer.model_max_length = 1024
model.config.max_length = 1024

# Define dataset class
class OCRDataset(Dataset):
    def __init__(self, text, labels):
        self.text = text
        self.labels = labels

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        try:
            text = str(self.text[idx])
            label = str(self.labels[idx])
            inputs = tokenizer(text, padding="max_length", truncation=True, max_length=1024)
            outputs = tokenizer(label, padding="max_length", truncation=True, max_length=1024)
            return {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "labels": outputs.input_ids,
                "decoder_attention_mask": outputs.attention_mask,
            }
        except Exception as e:
            logging.error(f"Error processing data at index {idx}: {e}")
            raise

# Create train and validation datasets
train_dataset = OCRDataset(
    text=train_df["ocr"].tolist(),
    labels=train_df["correct"].tolist()
)
valid_dataset = OCRDataset(
    text=eval_df["ocr"].tolist(),
    labels=eval_df["correct"].tolist()
)
logging.info("---- Data Processed ----")

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset
)

# Start training
logging.info("---- Training Started ----")
try:
    trainer.train()
    logging.info("---- Training Completed ----")
except Exception as e:
    logging.error(f"Training error: {e}")
    exit(1)

# Save the final model
try:
    trainer.save_model(training_args.output_dir)
    logging.info(f"Model saved to {training_args.output_dir}")
except Exception as e:
    logging.error(f"Error saving model: {e}")
    exit(1)

logging.info("----- Script Completed Successfully -----")
