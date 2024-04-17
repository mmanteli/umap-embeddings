import sys
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from numpy import array as array

# Embeddings file path
embedding_dir = sys.argv[1]
languages = sorted(sys.argv[2].split(","))
save_path = sys.argv[3] if len(sys.argv) > 3 else None
# downsampling if needed
#sample = int(sys.argv[3])*len(languages) if len(sys.argv) > 3 else len(languages)*2000
sample = int(sys.argv[4]) if len(sys.argv) > 4 else None #2000
seed = int(sys.argv[5]) if len(sys.argv) > 5 else None
save_path_suffix = "_seed_"+str(sys.argv[5]) if len(sys.argv) > 5 else ""


remove_small_class=True

split_dir = embedding_dir.split("/")
split_dir = [i for i in split_dir if i != ""]
lang_folder="-".join(languages)
model_name = split_dir[-1]   # for figure title
if save_path is None:
    save_path = "TEST_umap_figures/"+str(split_dir[-1])+save_path_suffix+"/"+lang_folder+"/"
print(f'Loading from {embedding_dir}', flush=True)
print(f'Saving to {save_path}', flush=True)
#x = input("THIS IS OLD NOW DO YOU WANT TO CONTINUE")

# UMAP settings
if seed is not None:
    reducer = umap.UMAP(random_state=seed)
else:
    reducer = umap.UMAP()
color_palette = sns.color_palette("Paired", 1000)

# Read data
dfs = []
for filename in os.listdir(embedding_dir):
    file = os.path.join(embedding_dir, filename)
    if any([l in filename for l in languages]):   # only take languages of intrest
        print(f'Reading {file}...', flush=True)
        # Get data
        df = pd.read_csv(file, sep="\t")
        if sample is not None:
            df = df.sample(n=sample)
        #df = df[np.random.choice(df.shape[0], sample, replace=False)]
        dfs.append(df)
        

# concat and sample
df = pd.concat(dfs)
del dfs
print("len: ",len(df), flush=True)
print(df.head(), flush=True)

# change from str to list
if False:#"xlmr" in model_name:      # This is stupid as hell but xlmr saved the results as a numpy tuple not as a list
    df["preds"] = df["preds"].apply(
        lambda a: eval(a)[0].tolist()
        )
else:
    df["preds"] = df["preds"].apply(
            lambda x: eval(x)
        )

df = df[df['preds'].apply(lambda x: len(x) < 2)]  # remove multilabel

if remove_small_class:
    print("Removing small classes")
    df = df[df['preds'].apply(lambda x: x not in [["MT"],["SP"],["LY"], []])]  # remove small class


df = df.explode("preds")  # Explode multilabels ??? This still needed???

# Values from string to list and flatten, get umap embeds
for column in ["embed_first","embed_half","embed_last"]:
    df[column] = df[column].apply(
        lambda x: np.array([float(y) for y in eval(x)[0]])
    )
    scaled_embeddings = StandardScaler().fit_transform(df[column].tolist())
    red_embedding = reducer.fit_transform(scaled_embeddings)
    df["x_"+column] = red_embedding[:, 0]
    df["y_"+column] = red_embedding[:, 1]


#fig, axes = plt.subplots(ncols=3, nrows=1)
for index, column in enumerate(["embed_first","embed_half","embed_last"]):
    plt.figure()
    plt.title(model_name+": "+column)
    print(column, flush=True)
    legend_elements = []
    for lang in languages:
        i = 0
        marker = {"en":"x", "fr":".", "zh":"1", "ur": "+", "fi":"*", "tr": "_", "fa": "d", "sv":"3"}[lang]
        legend_elements.append(Line2D([0],[0], color='w', markeredgecolor='black', markerfacecolor='black', marker=marker, label=lang, markersize=10))
        for label, group in df[df.lang==lang].groupby("preds"):
            if label==[] or label=="[]":
                continue
            plt.scatter(
                group["x_"+column],
                group["y_"+column],
                s=30,
                marker=marker,
                label=label,
                edgecolor="none",
                c=[color_palette[i]],
            )
            if lang==languages[-1]:
                legend_elements.append(Line2D([0],[0], color='w', markerfacecolor=color_palette[i], marker="s", label=label))
            i +=1

    lgnd = plt.legend(handles=legend_elements)
    for handle in lgnd.legend_handles:
        handle._sizes = [60]

    #plt.grid(True)
    #plt.show()
    Path(save_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path+"fig_"+column+".png")
    #axes[index] = plt.gcf()
    plt.close()

#fig.savefig("full.png")
