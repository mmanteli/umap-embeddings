from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import datasets
import json
import sys
import math
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.metrics import f1_score

def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--model_name', type=str, metavar='STR', default=None, required=True, 
                    choices = ["xlmr", "xlmr-long", "bge-m3","e5"],help='Which model to use.')
    ap.add_argument('--fold','--fold_number', type=int, metavar="INT", default=None,
                    help='Fold for xlmr and xlmr-long models')
    ap.add_argument('--data_name', type=str, metavar='STR', default=None, 
                    choices=["CORE", "cleaned", "register_oscar"],help='Which data to use.')
    ap.add_argument('--language','--lang', type=str, default=None, required=True, metavar='str',
                    help='which language to use.')
    ap.add_argument('--f1_limits', type=json.loads, metavar='ARRAY-LIKE', default=[0.3,0.65, 0.05],
                    help='[lower_limit, upper_limit, step] for f1 optimisation. 0.5 saved always.')
    ap.add_argument('--seed', type=int, metavar='INT', default=123,
                    help='Seed for reproducible outputs, like for sampling.')
    ap.add_argument('--save_path', type=str, metavar='DIR', default=None,
                    help='Where to save results. If none given, going to umap_embeddings/{data_name}/{model_name}/')
    return ap


# paths for models and data, e.g. data_path = data_dict("en")["CORE"] gives en-core
model_dict = lambda fold: {"xlmr":"/scratch/project_2009199/pytorch-registerlabeling/models/xlm-roberta-large/labels_upper/en-fi-fr-sv-tr_en-fi-fr-sv-tr/seed_42/fold_"+str(fold),
                        "xlmr-long":"/scratch/project_2009199/pytorch-registerlabeling/models/xlm-roberta-large_testing_longer_run/labels_upper/en-fi-fr-sv-tr_en-fi-fr-sv-tr/seed_42/fold_"+str(fold),
                        "e5":"/scratch/project_2009199/pytorch-registerlabeling/models/intfloat/multilingual-e5-large/labels_all/en-fi-fr-sv-tr_en-fi-fr-sv-tr/seed_42",
                        "bge-m3":"/scratch/project_2009199/bge-m3-multi-test"}

label_dict = {"xlmr": np.array(["MT","LY","SP","ID","NA","HI","IN","OP","IP"]), 
              "xlmr-long": np.array(["MT","LY","SP","ID","NA","HI","IN","OP","IP"]),
              "e5":np.array(["MT","LY","SP","ID","NA","HI","IN","OP","IP"]),
              "bge-m3":np.array(["HI","ID","IN","IP","LY","MT","NA","OP","SP"])}

data_dict = lambda lang: {"CORE": f'/scratch/project_2009199/sampling_oscar/final_core/{lang}.hf',
                          "cleaned": f'/scratch/project_2009199/sampling_oscar/final_cleaned/{lang}.hf',
                          "register_oscar": f'/scratch/project_2009199/sampling_oscar/final_reg_oscar/{lang}.hf',
                          "dirty": f'/scratch/project_2009199/sampling_oscar/final_dirty/{lang}.hf'}


options = argparser().parse_args(sys.argv[1:])
if options.model_name in ["xlmr", "xlmr-long"]:
    assert options.fold is not None, "No fold given for xmlr/xlmr-long"
options.model_path = model_dict(options.fold)[options.model_name]
options.data_path = data_dict(options.language)[options.data_name]
options.labels = label_dict[options.model_name]
if options.save_path is None:
    if options.model_name in ["xlmr", "xlmr-long"]:
        options.save_path = f'/scratch/project_2009199/umap-embeddings/model_embeds/{options.data_name}/{options.model_name}-fold-{options.fold}/'
    else:
       options.save_path = f'/scratch/project_2009199/umap-embeddings/model_embeds/{options.data_name}/{options.model_name}/' 


num_labels=len(options.labels)
label2id = {v:k for k,v in enumerate(options.labels)}
extract_labels = False if options.data_name=="cleaned" else True
base_model_name ="xlm-roberta-base"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

dataset = datasets.load_from_disk(options.data_path)
#dataset = dataset.filter(lambda example, idx: idx % 20 == 0, with_indices=True)
print(dataset)
model = AutoModelForSequenceClassification.from_pretrained(options.model_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model.to(device)


def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def predict(d, extract_labels=True):
    """
    Calculate sigmoids and get document averaged embeddings form 1st, last, middle and 3/4 model layers.
    Also, if labels 
    """
    with torch.no_grad():
        output = model(d["encoded"]["input_ids"].to(device), output_hidden_states=True)
    logits = output["logits"].cpu().tolist()
    sigm = np.array([sigmoid(v) for i,v in enumerate(logits[0]) if i < num_labels])
    hidden_states = output["hidden_states"]
    indices = np.array([0, len(hidden_states)//2, 3*len(hidden_states)//4, -1], dtype=int)
    embed = [torch.mean(hidden_states[i],axis=1).cpu().tolist() for i in indices]
    torch.cuda.empty_cache()
    if extract_labels:
        true_labels = np.zeros(num_labels, dtype=int)
        if d["labels"] is not None:
            if type(d["labels"])!= list:
                d["labels"] = d["labels"].split(" ")
            for l in d["labels"]:
                if l in label2id.keys():
                    true_labels[label2id[l]] = 1
        return {"id":d["id"], "lang":d["lang"], "prediction":sigm, "labels": d["labels"], "vec_labels": true_labels, "embed_first":embed[0], "embed_half":embed[1], "embed_last":embed[2]}
    return {"id":d["id"], "lang":d["lang"], "prediction":sigm, "embed_first":embed[0], "embed_half":embed[1], "embed_last":embed[2]}



def tokenize(d):
    if d["text"] is None:
        return tokenizer("", return_tensors='pt', truncation=True)
    return tokenizer(d["text"], return_tensors='pt', truncation=True)


dataset = dataset.map(lambda line: {"encoded": tokenize(line), "lang": options.language})
#dataset = dataset.map(lambda line: {"lang": options.language})
dataset = dataset.with_format("torch")


# this takes too much memory
#dataset = dataset.map(lambda line: predict(line))

# doing it with pandas instead
results = []
for d in tqdm(dataset["train"]):
    results.append(predict(d))

df = pd.DataFrame(results)
del dataset

# predictions for 0.5 threshold; applicable to all data
predictions = df["prediction"]
binary_predictions = [(prediction > 0.5).astype(int).tolist() for prediction in predictions]
preds = [options.labels[np.where(np.array(sublist) == 1)[0]].tolist() for sublist in binary_predictions]
df["preds"] = preds

# for data that has labels, calculate best f1 threshold
if extract_labels: #options.data_name != "cleaned":
    # get true labels
    true_labels = [i.tolist() for i in df["vec_labels"]]
    # init saving
    best_f1=0
    best_threshold = None
    best_predictions = []
    
    for threshold in np.arange(options.f1_limits[0],options.f1_limits[1],options.f1_limits[2]):
        binary_predictions = [(prediction > threshold).astype(int).tolist() for prediction in predictions]
        f1 = f1 = f1_score(y_true=true_labels, y_pred=binary_predictions, average="micro")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_predictions = binary_predictions
    #print(best_f1, np.round(best_threshold, decimals=1))

    # take the labels from the best saved predictions
    preds = [options.labels[np.where(np.array(sublist) == 1)[0]].tolist() for sublist in best_predictions]
    df["preds_"+str(np.round(best_threshold, decimals=1))] = preds


#print(df)
# save results
Path(options.save_path).mkdir(parents=True, exist_ok=True)
df.to_csv(str(options.save_path)+str(options.language)+"_embeds.tsv", sep="\t", header=True)
