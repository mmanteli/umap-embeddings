from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import datasets
import json
import sys
import pandas as pd
from tqdm import tqdm
from pathlib import Path

lang = sys.argv[1]

save_path = sys.argv[2] if len(sys.argv) > 2 else "/scratch/project_2009199/embeddings-and-umap/model_embeds/bge-m3/"

base_model_name ="xlm-roberta-base"
model_name = "/scratch/project_2009199/bge-m3-multi-test"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

dataset = datasets.load_from_disk("/scratch/project_2009199/sampling_oscar/final/"+str(lang)+".hf")
dataset = dataset.filter(lambda example, idx: idx % 10 == 0, with_indices=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model.to(device)

# originally using different data where downsampling was language dependent
# downsample = {"en":25, "fr": 48, "zh": 18}
# dataset = datasets.load_dataset("TurkuNLP/register_oscar", data_files={lang:f'{lang}/{lang}_00000.jsonl.gz'}, cache_dir="/scratch/project_2009199/cache")
# dataset = dataset.filter(lambda example, idx: idx % downsample[lang] == 0, with_indices=True)

labels = np.array(["HI","ID","IN","IP","LY","MT","NA","OP","SP"])
def predict(d,lang):
    with torch.no_grad():
        output = model(d["encoded"]["input_ids"].to(device), output_hidden_states=True)
    #logits = output["logits"].cpu()
    #pred = [labels[i] for i in np.where(logits > 0.5)[1] if i < 9]  # small labels (>=9) get the heck out
    #hidden_states = output["hidden_states"]
    logits = output["logits"].cpu().tolist()
    sigm = np.array([sigmoid(i) for i in logits[0] if i < 9])   # small labels (>=9) get the heck out
    pred = [labels[i] for i in np.where(sigm > 0.5)]
    indices = np.array([0, len(hidden_states)//2, -1], dtype=int)
    embed = [torch.mean(hidden_states[i],axis=1).cpu().tolist() for i in indices]
    torch.cuda.empty_cache()
    return {"id": d["id"], "lang":lang, "labels": d["labels"],"preds": pred, "embed_first":embed[0], "embed_half":embed[1], "embed_last":embed[2]}
    #return [d["id"],d["labels"],pred,embed]


def tokenize(d):
    return tokenizer(d["text"], return_tensors='pt', truncation=True)


dataset = dataset.map(lambda line: {"encoded": tokenize(line)})
dataset = dataset.with_format("torch")

#for d in dataset["en"]:
#    print(json.dumps(predict(d)))
results = []
for d in tqdm(dataset["train"]):
    results.append(predict(d,lang))

Path(save_path).mkdir(parents=True, exist_ok=True)

df = pd.DataFrame(results)
df.to_csv(save_path+lang+"_embeds.tsv", sep="\t", header=True)
