import pandas as pd
import sqlite3
import evaluate
import os
import json
import matplotlib.pyplot as plt
import seaborn
from typing import Set
import numpy as np
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
import torch
from torch.utils.data import Dataset

def execute_query(query, print_padding, verbose=True):
    try:
        conn = sqlite3.connect("data\\reza_data_cars.db")
        cursor = conn.cursor()
        cursor.execute(query)
        if verbose:
            print(f"{'SQL query:':<{print_padding}}{query}")
            print(f"{'Query results:':<{print_padding}}{cursor.fetchall()}")
        return cursor.fetchall()
    except sqlite3.Error as error:
        print(f"Error executing sql {error}")
        
    finally:
        conn.close()

# Load the tokenizer and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('cssupport/t5-small-awesome-text-to-sql').to(device)

# fine-tune the model
# t5 fine-tuning article https://medium.com/nlplanet/a-full-guide-to-finetuning-t5-for-text2text-and-building-a-demo-with-streamlit-c72009631887


model.eval()

def generate_sql(input_prompt):
    """Generate SQL query from natural language input."""
    inputs = tokenizer(input_prompt, padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

padding = 30
print("Good examples:\n")
# example 
natural_language_query = "How many makers are there?"
input_prompt = f"""tables:
CREATE TABLE "car_makers" ( 
	Id INTEGER PRIMARY KEY, 
	Maker TEXT, 
	FullName TEXT, 
	Country TEXT,
	FOREIGN KEY (Country) REFERENCES countries(CountryId)
)
### Write an SQL query to answer the question:
{natural_language_query}"""

generated_sql = generate_sql(input_prompt)
print(f"{'Original query:':<{padding}}{natural_language_query}\n{'Generated SQL:':<{padding}}{generated_sql}")
execute_query(generated_sql, padding)

# example 
natural_language_query = "Who is the heaviest car?"
input_prompt = f"""tables:
CREATE TABLE "cars_data" (
	Id INTEGER PRIMARY KEY, 
	MPG TEXT, 
	Cylinders INTEGER, 
	Edispl REAL, 
	Horsepower TEXT, 
	Weight INTEGER, 
	Accelerate REAL, 
	Year INTEGER,
	FOREIGN KEY (Id) REFERENCES car_names (MakeId)
)
CREATE TABLE "car_names" ( 
	MakeId INTEGER PRIMARY KEY, 
	Model TEXT, 
	Make TEXT,
	FOREIGN KEY (Model) REFERENCES model_list (Model)
)
### Write an SQL query to answer the question:
{natural_language_query}"""

generated_sql = generate_sql(input_prompt)
print(f"\n{'Original query:':<{padding}}{natural_language_query}\n{'Generated SQL:':<{padding}}{generated_sql}")
execute_query(generated_sql, padding)

# example 
natural_language_query = "How many makers are there?"
input_prompt = f"""tables:
CREATE TABLE "model_list" (
	Model TEXT UNIQUE
)
### Write an SQL query to answer the question:
{natural_language_query}"""

generated_sql = generate_sql(input_prompt)
print(f"\n{'Original query:':<{padding}}{natural_language_query}\n{'Generated SQL:':<{padding}}{generated_sql}")
execute_query(generated_sql, padding)

# re-pull model to ensure I'm running against the right model in noteboook
# Use pre-trained model to test output of training data before fine-tuning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('cssupport/t5-small-awesome-text-to-sql').to(device)

match_count = 0
for example in training_data:
    input_prompt = example["input"]
    expected_sql = example["target"]
    generated_sql = generate_sql(input_prompt)
    generated_results = execute_query(generated_sql, padding, verbose=False)
    print(f"\n{'Generated SQL:':<{padding}}{generated_results}")
    expected_sql_results = execute_query(expected_sql, padding, verbose=False)
    print(f"{'Expected SQL:':<{padding}}{expected_sql_results}")
    print("**" * 50)
    if generated_results == expected_sql_results:
        match_count += 1

print(f"Total Matches: {match_count} out of {len(training_data)}")

class TextToSQLDataset(Dataset):
    def __init__(self, tokenizer, data, max_length=512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = self.data[idx]['input']
        target_text = self.data[idx]['target']
        
        # Tokenize inputs and targets
        inputs = self.tokenizer(input_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")
        targets = self.tokenizer(target_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")
        
        labels = targets.input_ids.squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100 # ! T5 expects padding in labels to be set to -100

        
        return {
            'input_ids': inputs.input_ids.squeeze(0),
            'attention_mask': inputs.attention_mask.squeeze(0),
            'labels': labels
        }
    
from sklearn.model_selection import train_test_split

train_examples, val_examples = train_test_split(training_data, test_size=0.1, random_state=123)
train_dataset = TextToSQLDataset(tokenizer, train_examples)
val_dataset   = TextToSQLDataset(tokenizer, val_examples)

meteor_metric = evaluate.load("meteor")
bleu_metric = evaluate.load("bleu")


def hallucination_rate(pred: str, schema_tokens: Set[str], reference: str):
    # tokens in pred not in schema or reference
    pred_tokens = set(pred.split())
    ref_tokens = set(reference.split())
    extra = pred_tokens - schema_tokens - ref_tokens
    return len(extra) / len(pred_tokens)


def compute_metrics(eval_pred):
    # With Seq2SeqTrainer & predict_with_generate=True,
    # eval_pred.predictions are already token IDs from .generate()
    preds, labels = eval_pred.predictions, eval_pred.label_ids
    
    # Decode predictions & labels
    decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
    # replace -100 in the labels as pad_token_id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # clean whitespace
    preds_clean = [p.strip() for p in decoded_preds]
    labs_clean  = [[l.strip()] for l in decoded_labels]

    # BLEU & METEOR
    bleu   = bleu_metric.compute(predictions=preds_clean, references=labs_clean)["bleu"]
    meteor = meteor_metric.compute(predictions=preds_clean, references=labs_clean)["meteor"]

    # Exact match
    exact = sum(p == l[0] for p,l in zip(preds_clean, labs_clean)) / len(labs_clean)

    schema_tokens = {
        "car_makers",
        "car_makers.Id",
        "Id",
        "car_makers.Maker",
        "Maker",
        "car_makers.FullName",
        "FullName",
        "car_makers.Country",
        "Country",
        "car_names",
        "car_names.MakeId",
        "MakeId",
        "car_names.Model",
        "Model",
        "car_names.Make",
        "Make",
        "cars_data",
        "cars_data.Id",
        "Id",
        "cars_data.MPG",
        "MPG",
        "cars_data.Cylinders",
        "Cylinders",
        "cars_data.Edispl",
        "Edispl",
        "cars_data.Horsepower",
        "Horsepower",
        "cars_data.Weight",
        "Weight",
        "cars_data.Accelerate",
        "Accelerate",
        "cars_data.Year",
        "Year",
        "continents",
        "continents.ContId",
        "ContId",
        "continents.Continent",
        "Continent",
        "continents.countries",
        "countries",
        "continents.CountryId",
        "CountryId",
        "continents.CountryName",
        "CountryName",
        "model_list",
        "model_list.ModelId",
        "ModelId",
        "model_list.Maker",
        "Maker",
        "model_list.Model",
        "Model",
    }  # column names and table names for tokens to validate against hallucinations
    rates = [
        hallucination_rate(p, schema_tokens, l)
        for p, l in zip(decoded_preds, decoded_labels)
    ]
    hallu = sum(rates) / len(rates)

    return {
        "bleu": float(bleu),
        "meteor": float(meteor),
        "exact_match": float(exact),
        "hallucination_rate": float(hallu),
    }

# re-pull model to ensure I'm running against the right model in noteboook
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('cssupport/t5-small-awesome-text-to-sql').to(device)

training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    num_train_epochs=50, # Will be overfitting the model
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=3e-4,
    weight_decay=0.0,
    logging_dir='./logs',
    # logging_steps=1, # commented out for smaller export
    save_strategy="no",
    remove_unused_columns=False,
    report_to="none",
    predict_with_generate=True,
)


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

# Save the model
model.save_pretrained('./fine-tuned-model')
tokenizer.save_pretrained('./fine-tuned-model')

# Use the newly fine-tuned model to generate SQL queries
# I am using the same questions as what was provided in the fine-tuning data
# to see if the fine-tuned model can generate the correct SQL queries after the training.
fine_tuned_model_path = './fine-tuned-model'
tokenizer = T5Tokenizer.from_pretrained(fine_tuned_model_path)
model = T5ForConditionalGeneration.from_pretrained(fine_tuned_model_path).to(device)

def generate_sql_fine_tuned(input_prompt):
    """Generate SQL query from natural language input using the fine-tuned model."""
    inputs = tokenizer(input_prompt, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=256)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
 
# Print the generated SQL query's results and compare them with the expected query's results from training
match_count = 0
for example in training_data:
    input_prompt = example["input"]
    expected_sql = example["target"]
    generated_sql = generate_sql_fine_tuned(input_prompt)
    generated_results = execute_query(generated_sql, padding, verbose=False)
    expected_sql_results = execute_query(expected_sql, padding, verbose=False)
    if generated_results == expected_sql_results:
        match_count += 1
    else:
        print(f"\n{'Generated SQL:':<{padding}}{generated_sql}")
        print(f"{'Expected SQL:':<{padding}}{expected_sql}")

print(f"Total Matches: {match_count} out of {len(training_data)}")