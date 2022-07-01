#load dependencies 

from pip._internal import main as pipmain

pipmain(['install', 'transformers'])
pipmain(['install', 'datasets'])
pipmain(['install', 'huggingface'])

import pandas as pd
import numpy as np
import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from datasets import Dataset
from transformers import TrainingArguments
from transformers import Trainer 


#eval metrics
from sklearn.metrics import mean_absolute_error, accuracy_score

#define functions

# make labels
def make_labels():

    labels = df_osdg['target'].unique()
    labels.sort()
    num_labels = len(labels)
    id2label = {idx:label for idx, label in enumerate(labels)}
    label2id = {label:idx for idx, label in enumerate(labels)}

    return labels, num_labels, id2label, label2id

# set-up tokenizer
def tokenize_function(examples):

    return tokenizer(examples["text_clean"], padding="max_length", truncation=True,max_length=135)

# define metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return {"Acc": accuracy_score(labels, predictions)}

# map labels
def map_labels(example):
    # Shift labels to start from 0
    label_id = label2id[example["target"]]
    return {"labels": label_id, "label_name": id2label[label_id]}

if __name__ == '__main__':
    
    #load data
    df_osdg = pd.read_csv("../data/processed/data_transformer.csv")
    df_osdg['target'] = df_osdg['target'].astype(str)
    dataset =  Dataset.from_pandas(df_osdg)

    # make labels
    labels, num_labels, id2label, label2id = make_labels()
    dataset = dataset.map(map_labels)

    # split to train and test data
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.train_test_split(test_size=0.1)

    model_checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    print(tokenizer.vocab_size)

    # set up tokenizer
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    label2id = {v:k for k,v in id2label.items()}

    encoded_str = tokenizer("Today I'm giving an NLP workshop at MLT")
    print(encoded_str)

    # load pre-trained model
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels, label2id=label2id, id2label=id2label)

    model_name = model_checkpoint.split("/")[-1]
    batch_size = 32
    num_train_epochs = 7
    logging_steps = len(tokenized_datasets["train"]) // (batch_size * num_train_epochs)

    args = TrainingArguments(
        output_dir=f"{model_name}-finetuned-sdg",
    evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=6e-7,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        #weight_decay=0.005,
        logging_steps=logging_steps,
        #push_to_hub=True,
    )

    # specify trainer
    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # train
    trainer.train()
    trainer.save_model()

    print("done!")