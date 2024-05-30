import pandas as pd

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import f1_score, confusion_matrix
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn



class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def read_dataset():
    train = pd.read_csv("spam_model_selection/dataset/train.csv")
    dev = pd.read_csv("spam_model_selection/dataset/dev.csv")
    test = pd.read_csv("spam_model_selection/dataset/test.csv")

    return train, dev, test


def prepare_model_for_binary():
    tokenizer = AutoTokenizer.from_pretrained("uitnlp/visobert", use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained("uitnlp/visobert", num_labels=2)
    return tokenizer, model


def train_model_binary(train_raw, dev_raw):
    tokenizer, model = prepare_model_for_binary()

    train_text = [str(x) if x is not None else '' for x in train_raw['Comment'].values]
    dev_text = [str(x) if x is not None else '' for x in dev_raw['Comment'].values]

    train_encodings = tokenizer(train_text, truncation=True, padding=True, max_length=200)
    dev_encodings = tokenizer(dev_text, truncation=True, padding=True, max_length=200)

    train_dataset = BuildDataset(train_encodings, train_raw['Label'].values)
    dev_dataset = BuildDataset(dev_encodings, dev_raw['Label'].values)

    MODEL_PATH = './spam_model_selection/model/'
    training_args = TrainingArguments(
        output_dir=MODEL_PATH,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset
    )
    trainer.train()
    trainer.save_model(MODEL_PATH)

    return trainer, tokenizer

def evaluate_model_binary(test_raw, trainer, tokenizer):
    test_text = [str(x) if x is not None else '' for x in test_raw['Comment']]

    test_encodings = tokenizer(test_text, truncation=True, padding=True, max_length=100)
    test_dataset = BuildDataset(test_encodings, test_raw['Label'].values)

    y_pred_classify = trainer.predict(test_dataset)

    y_pred = np.argmax(y_pred_classify.predictions, axis=-1)
    y_true = test_raw['Label'].values

    cf2 = confusion_matrix(y_true, y_pred)
    print(cf2)

    evaluation = f1_score(y_true, y_pred, average='macro')
    print("F1 - macro: " + str(evaluation))

    # Show out the confusion matrix
    df_cm2 = pd.DataFrame(cf2, index = ["no-spam","spam"],
                    columns = ["no-spam","spam"])
    plt.clf()
    sn.heatmap(df_cm2, annot=True, cmap="Greys",fmt='g', cbar=True, annot_kws={"size": 30})


#Khai báo trước label, model và tokenize cho mô hìm dự báo spam:
idx2label = {
        0: "No Spam",
        1: "Spam"
    }
model_predict_spam = AutoModelForSequenceClassification.from_pretrained('./spam_model_selection/model', num_labels = 2)
tokenizer_predict_spam = AutoTokenizer.from_pretrained("uitnlp/visobert",use_fast=False)
# Di chuyển mô hình đến thiết bị (GPU nếu có)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_spam_binary(text_raw):
     # Mã hóa dữ liệu đầu vào
    encoding = tokenizer_predict_spam(text_raw, truncation=True, padding=True, max_length=200, return_tensors='pt')
    
    # Di chuyển dữ liệu đến thiết bị (GPU nếu có)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Tắt gradient calculation
    with torch.no_grad():
        # Dự đoán
        outputs = model_predict_spam(input_ids, attention_mask=attention_mask)
        predictions = outputs.logits
    
    # Tìm nhãn có xác suất cao nhất
    y_pred = torch.argmax(predictions, axis=-1).item()
    
    return idx2label[y_pred]








