#load dependencies 

import pandas as pd
import numpy as np
import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from datasets import Dataset
from transformers import TrainingArguments
from transformers import Trainer 

from argparse import ArgumentParser
from sklearn.metrics import mean_absolute_error, accuracy_score
import torch 
from torch import cuda, true_divide
from accelerate import Accelerator

from datasets import load_metric

accelerator = Accelerator()
device = accelerator.device

# this is for tracking the experiment. Comment it out if you do not wish to log the training or you do not have
# a wandb account. We will also use wandb to apply hyperparameter tuning. 
# Note that the API is not pushed on github for security reasons. Either create a txt file with your wandb API key
# and import it or directly specify the api key under key=. 
import wandb
from pathlib import Path
wandb_api_key = Path('data/wandb_api_key.txt').read_text()

wandb.login(key=wandb_api_key)
#git wandb.init(project="sdg-classifier-roberta-base")

#define functions

# set-up tokenizer
def tokenize_function(examples):

    return tokenizer(examples["text_clean"],  truncation=True, padding=True)


def compute_metrics_fn(eval_preds):
  metrics = dict()
  
  accuracy_metric = load_metric('accuracy')
  precision_metric = load_metric('precision')
  recall_metric = load_metric('recall')
  f1_metric = load_metric('f1')


  logits = eval_preds.predictions
  labels = eval_preds.label_ids
  preds = np.argmax(logits, axis=-1)  
  
  metrics.update(accuracy_metric.compute(predictions=preds, references=labels))
  metrics.update(precision_metric.compute(predictions=preds, references=labels, average='weighted'))
  metrics.update(recall_metric.compute(predictions=preds, references=labels, average='weighted'))
  metrics.update(f1_metric.compute(predictions=preds, references=labels, average='weighted'))


  return metrics

def model_init():
    model_checkpoint = "roberta-base"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels, 
        label2id=label2id, 
        id2label=id2label
    )
    
    return model 

# define hyperparameters tunig - set sweep config for wanb

# method
sweep_config = {
    'method': 'random'
}

# hyperparameters
parameters_dict = {
    'epochs': {
        'value': 1
        },
    'batch_size': {
        'values': [8, 16, 32, 64]
        },
    'learning_rate': {
        'distribution': 'log_uniform_values',
        'min': 1e-5,
        'max': 1e-3
    },
    'weight_decay': {
        'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    },
}


sweep_config['parameters'] = parameters_dict
sweep_id = wandb.sweep(sweep_config, project='sdg-classifier-roberta-base')

def train(config=None):

    with wandb.init(config=config):

        # set sweep configuration
        config = wandb.config

        # set training arguments
        training_args = TrainingArguments(
            output_dir="./models/"+f"{model_name}-finetuned-sdg",
            report_to='wandb',  # Turn on Weights & Biases logging
            num_train_epochs=config.epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=16,
            save_strategy='epoch',
            evaluation_strategy='epoch',
            logging_strategy='epoch',
            load_best_model_at_end=True,
            remove_unused_columns=False,
        )

        # define training loop
        trainer = Trainer(
            #model = model,
            model_init=model_init,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_fn
        )
    
        # start training loop
        trainer.train()
        trainer.save_model()

if __name__ == '__main__':

    #load data
    df_osdg = pd.read_csv("./data/processed/data_transformer.csv")
    df_osdg['target'] = df_osdg['target'].astype(int)
    dataset =  Dataset.from_pandas(df_osdg)

    # split to train and test data
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.train_test_split(test_size=0.2)

    model_name = "roberta-base"
    # load pre-trained model
    #model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, label2id=label2id, id2label=id2label)
    model_name = model_name.split("/")[-1]
    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, padding=True)
    # set up tokenizer
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    label2id = {v:k for k,v in id2label.items()}

    wandb.agent(sweep_id, train, count=20)

    print("done!")