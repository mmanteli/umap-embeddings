from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import datasets
import json
import sys
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import classification_report

lang = sys.argv[1]
fold = sys.argv[2]
save_path = sys.argv[3] if len(sys.argv) >= 4 else "/scratch/project_2009199/embeddings-and-umap/reports/xlmr-fold-"+str(fold)+"/"

base_model_name ="xlm-roberta-base"
model_name = "/scratch/project_2009199/pytorch-registerlabeling/models/xlm-roberta-large/labels_upper/en-fi-fr-sv-tr_en-fi-fr-sv-tr/seed_42/fold_"+str(fold)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

dataset = datasets.load_from_disk("/scratch/project_2009199/sampling_oscar/final/"+str(lang)+".hf")
dataset = dataset.filter(lambda example, idx: idx % 10 == 0, with_indices=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model.to(device)

import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# originally using different data where downsampling was language dependent
# downsample = {"en":25, "fr": 48, "zh": 18}
# dataset = datasets.load_dataset("TurkuNLP/register_oscar", data_files={lang:f'{lang}/{lang}_00000.jsonl.gz'}, cache_dir="/scratch/project_2009199/cache")
# dataset = dataset.filter(lambda example, idx: idx % downsample[lang] == 0, with_indices=True)

labels = np.array(["MT","LY","SP","ID","NA","HI","IN","OP","IP"])
def predict(d,lang):
    with torch.no_grad():
        output = model(d["encoded"]["input_ids"].to(device))
    logits = output["logits"].cpu().tolist()
    #print(logits)
    sigm = [sigmoid(i) for i in logits[0]]
    #pred = [labels[i] for i in np.where(logits > 0.5)[1]]
    pred = [int(i>0.5) for i in sigm]
    torch.cuda.empty_cache()
    return pred


def tokenize(d):
    return tokenizer(d["text"], return_tensors='pt', truncation=True)


dataset = dataset.map(lambda line: {"encoded": tokenize(line)})
dataset = dataset.with_format("torch")

preds = []
trues = []
for d in tqdm(dataset["train"]):
    label = d["labels"]
    true = [int(l in label) for l in labels]
    #print(true)
    trues.append([int(l in label) for l in labels])
    preds.append(predict(d,lang))

Path(save_path).mkdir(parents=True, exist_ok=True)
report = classification_report(trues, preds, target_names=labels)#, output_dict=True)
#print(report)
with open(save_path+lang+"_report.txt", "w") as f:
    f.write(report)
#df = pd.DataFrame(report).transpose()
#df.to_csv(save_path+lang+"_report.txt", sep="\t")
