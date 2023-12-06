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

from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.commands.user import login

api = HfApi()
login(token = Path('data/huggingface_api_key.txt').read_text())

# this is for tracking the experiment. Comment it out if you do not wish to log the training or you do not have
# a wandb account. We will also use wandb to apply hyperparameter tuning. 
# Note that the API is not pushed on github for security reasons. Either create a txt file with your wandb API key
# and import it or directly specify the api key under key=. 
import wandb
wandb_api_key = Path('data/wandb_api_key.txt').read_text()

wandb.login(key=wandb_api_key)
#git wandb.init(project="sdg-classifier-roberta-base")

#define functions

# make labels
def make_labels():
    """
    mapping the labels. This is specifically important if we want to preserve original label names. 
    """
    labels = df_osdg['target'].unique()
    labels.sort()
    num_labels = len(labels)
    id2label = {idx:label for idx, label in enumerate(labels)}
    label2id = {label:idx for idx, label in enumerate(labels)}

    return labels, num_labels, id2label, label2id

# set-up tokenizer
def tokenize_function(examples):
    """
    our tokenizer function. Here we specify max lenght of 135 tokens. This is an individual decision and depends on the downstream tasks and the available data at hand.
    Since our dataset is already processed to a specific lenght during pre-processing we would not need to specify a value here and just set padding as True. 

    Note that the tokenizer function returns the original columns as well. These have to be removed before fed into the transformer model for fine-tuning.
    """
    return tokenizer(examples["text_clean"], padding="max_length", truncation=True, max_length=135)

def compute_metrics_fn(eval_preds):
    """
    load different metrics. The default metric is accuracy score. However, especially with inbalanced datasets we want to obtain more metrics for evualuation.
    We will mostly look at the weighted F1 score as the harmonic mean of precision and recall. 
    """
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

# map labels
def map_labels(example):
    # Shift labels to start from 0
    label_id = label2id[example["target"]]
    return {"labels": label_id, "label_name": id2label[label_id]}

def model_init():
    """
    the function that is called during the training sweeps to re-initialise the model of our choice.
    """
    model_checkpoint = "roberta-base"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels, 
        label2id=label2id, 
        id2label=id2label
    )
    
    return model 

# hyperparameters
"""
here we specify the hyperparameters we want to test during the sweeps. There is no real right or wrong here, but it makes sense to orient at the default values and
deviate a bit to both sides. 
"""
# method
sweep_config = {
    'method': 'random'
}

parameters_dict = {
    'epochs': {
        'values':[2, 3, 4, 5, 6]
        },
    'batch_size': {
        'values': [8, 16, 32, 64]
        },
    'learning_rate': {
        'values': [3e-4, 1e-4, 5e-5, 3e-5]
    },
    'weight_decay': {
        'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    },
}

sweep_config['parameters'] = parameters_dict
sweep_id = wandb.sweep(sweep_config, project='sdg-classifier-roberta-July23')

def train(config=None):

    with wandb.init(config=config):

        # set sweep configuration
        config = wandb.config

        # set training arguments
        training_args = TrainingArguments(
            output_dir="./models/"+f"{model_name}-finetuned-roberta-sdg-July23",
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
            #data_collator=collate_fn,
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
    df_osdg['target'] = df_osdg['target'].astype(str)
    dataset =  Dataset.from_pandas(df_osdg)

    # make labels
    labels, num_labels, id2label, label2id = make_labels()
    dataset = dataset.map(map_labels)
    label2id = {v:k for k,v in id2label.items()}

    # split to train and test data
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.train_test_split(test_size=0.2)

    model_name = "bert-base-uncased"
    # load pre-trained model
    #model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, label2id=label2id, id2label=id2label)
    model_name = model_name.split("/")[-1]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # set up tokenizer
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    # remove unwanted columns 
    tokenized_datasets = tokenized_datasets.remove_columns(["text_clean", "target", "label_name"])

    # start sweep 
    wandb.agent(sweep_id, train, count=12)

    print("done!")