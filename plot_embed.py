import sys
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
import umap
import re
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from numpy import array as array
import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

all_labels = ["HI","ID","IN","IP","LY","MT","NA","OP","SP"]
big_labels = ["HI","ID","IN","IP","NA","OP"]



def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--embeddings', type=str, default=None, required=True, metavar='DIR',
                    help='Path to directory where precalculated embeddings are. Lang name assumed in the filename.')
    ap.add_argument('--languages','--langs','--language','--lang', type=str, metavar='LIST', default=None, required=True,
                    help='Which languages to download from --embeddings, give as: en,fr,zh.')
    ap.add_argument('--labels', type=json.loads, metavar='ARRAY-LIKE', default=all_labels,
                    help='Labels. Give as: \'["IN","NA"]\'. Others discarded. Works only with --remove_multilabel.')
    ap.add_argument('--sample', type=int, metavar='INT', default=2000,
                    help='How much to sample from each language.')
    ap.add_argument('--model_name', type=str, metavar='STR', default=None, required=True,
                    help='Name of the model for plot titles.')
    ap.add_argument('--data_name', type=str, metavar='STR', default=None,
                    help='If given, added to plot title.')
    ap.add_argument('--use_column', '--column', type=str, metavar='STR', default="preds", choices=["preds","labels", "preds_best"],
                    help='Which column containing labels to use, "preds" "labels", "preds_best".')
    ap.add_argument('--remove_multilabel', type=str, metavar='BOOL', default="True",
                    help='Remove docs with multilabel predictions.')
    ap.add_argument('--n_neighbors', type=int, metavar='INT', default=15,
                    help='How many neighbors for UMAP.')
    ap.add_argument('--seed', type=int, metavar='INT', default=None,
                    help='Seed for reproducible outputs. Default=None, UMAP runs faster with no seed.')
    ap.add_argument('--save_path', type=str, metavar='DIR', default=None,
                    help='Where to save results. Defaults to sameas --embeddings with model_embeds => umap_figures and langs added.')
    return ap
    


options = argparser().parse_args(sys.argv[1:])
languages_as_list = options.languages.split(",")
languages_as_list.sort()
options.labels.sort()
options.remove_multilabel = bool(eval(options.remove_multilabel))
if options.embeddings[-1] == "/":
    options.embeddings = options.embeddings[:-1]
if options.save_path is None:
    lang_folder = "-".join(languages_as_list)
    label_folder = "" if options.labels==all_labels else "-large-labels" if options.labels==big_labels else "-"+"-".join(options.labels)
    options.save_path = options.embeddings.replace("model_embeds", "umap-figures") + "/"+lang_folder+label_folder+"/"
wrt_column = options.use_column
print("-----------------INFO-------------------", flush=True)
print(f'Loading from: {options.embeddings+"/"}', flush=True)
print(f'Saving to {options.save_path}', flush=True)
print(f'Plotting based on column: {wrt_column}', flush=True)
print("-----------------INFO-------------------", flush=True)

# UMAP settings
if options.seed is not None:
    reducer = umap.UMAP(random_state=options.seed, n_neighbors=options.n_neighbors)
else:
    reducer = umap.UMAP(n_neighbors=options.n_neighbors)



def do_sampling(df, n, r=1):
    """
    Sampling that also accepts None as a sample, so that less if-conditions are needed.
    r = ratio: how many times the sampling constant n we sample. To facilitate faster processing,
    i.e.:
        1. sample to 2*n
        2. process (and remove less than n instances)
        3. sample to n
    """
    if n is None:
        return df
    try:
        return df.sample(n=r*n)
    except:
        if r==1:  # only report at the final step
            print(f'Unable to sample to {n} < len(df) = {len(df)}.', flush=True)
        return df


# Read data
dfs = []
for filename in os.listdir(options.embeddings):
    file = os.path.join(options.embeddings, filename)
    if any([l in filename.replace(".tsv","") for l in options.languages]):   # only take languages of intrest
        print(f'Reading {file}...', flush=True)
        df = pd.read_csv(file, sep="\t")
        # rename the best f1 column from preds_{value} to preds_best
        df.rename(columns=lambda x: re.sub('_0.*','_best',x), inplace=True)
        # sample down to 2*sample to make running things faster
        df = do_sampling(df,options.sample, r=2)
        # from str to list, wrt_column contains the column we are interested in
        print(df)
        print(df.columns)
        try:
            df[wrt_column] = df[wrt_column].apply(
                lambda x: eval(x)
            )
        except:   # CORE labels separated by white space, and NA maps to NaN in eval
            df[wrt_column] = df[wrt_column].apply(
                lambda x: ["NA" if i=="nan" else str(i) for i in str(x).split(" ")]
            )
        # remove multilabel
        print(len(df))
        if options.remove_multilabel:
            df = df[df[wrt_column].apply(lambda x: len(x) < 2 and len(x) > 0)]
            if options.labels != all_labels:     # some classes need to be removed
                print(f'Removing all classes except {options.labels}', flush=True)
                df = df[df[wrt_column].apply(lambda x: x not in options.labels)]  # remove unwanted classes
        else:
            # only remove empty
            df = df[df[wrt_column].apply(lambda x: len(x) > 0)]
        # sample to options.sample
        print(len(df))
        df = do_sampling(df,options.sample) 
        dfs.append(df)


# concat and remove multilabel
df = pd.concat(dfs)
del dfs
print("len: ",len(df), flush=True)
print(df.head(), flush=True)
df = df.explode(wrt_column)  # Does something and is needed

# Values from string to list and flatten, get umap embeds
for column in ["embed_first","embed_half","embed_last"]:
    df[column] = df[column].apply(
        lambda x: np.array([float(y) for y in eval(x)[0]])
    )
    scaled_embeddings = StandardScaler().fit_transform(df[column].tolist())
    red_embedding = reducer.fit_transform(scaled_embeddings)
    df["x_"+column] = red_embedding[:, 0]
    df["y_"+column] = red_embedding[:, 1]



embed_name = {"embed_first":"first_layer","embed_half":"halfway_layer","embed_last":"last_layer"}

# color palette
color_palette = sns.color_palette("Paired", len(options.labels))
for index, column in enumerate(["embed_first","embed_half","embed_last"]):
    plt.figure()
    plt.figure(figsize=(10,7.5))
    if options.data_name is not None:
        plt.title(options.model_name+": "+embed_name[column]+", data:"+options.data_name)
    else:
         plt.title(model_name+": "+embed_name[column])
    print(column, flush=True)
    legend_elements = []
    plt.xlabel("wrt. "+wrt_column+",n_neigh = "+str(options.n_neighbors))
    for lang in languages_as_list:
        i = 0
        marker = ","#{"en":"x", "fr":".", "zh":"1", "ur": "+", "fi":"*", "tr": "_", "fa": "d", "sv":"3"}[lang]
        #legend_elements.append(Line2D([0],[0], color='w', markeredgecolor='black', markerfacecolor='black', marker=marker, label=lang, markersize=10))
        for label, group in df[df.lang==lang].groupby(wrt_column):
            if label==[] or label=="[]":
                continue
            plt.scatter(
                group["x_"+column],
                group["y_"+column],
                s=5,
                marker=marker,
                label=label,
                edgecolor="none",
                c=[color_palette[i]],
                alpha=0.3,
            )
            if lang==languages_as_list[-1]:  #for last language add the legend
                legend_elements.append(Line2D([0],[0], color='w', markerfacecolor=color_palette[i], marker="s", label=label))
            i +=1

    lgnd = plt.legend(handles=legend_elements)
    for handle in lgnd.legend_handles:
        handle._sizes = [60]

    Path(options.save_path).mkdir(parents=True, exist_ok=True)
    
    fig_save_path = options.save_path+"fig_"+embed_name[column]+"_"+wrt_column
    if options.seed != None:
        fig_save_path += "_seed_"+str(options.seed)
        
    plt.savefig(fig_save_path+".png")
    plt.close()


color_palette = sns.color_palette("hls", len(languages_as_list))
for index, column in enumerate(["embed_first","embed_half","embed_last"]):
    plt.figure()
    plt.figure(figsize=(10,7.5))
    if options.data_name is not None:
        plt.title(options.model_name+": "+embed_name[column]+", data:"+options.data_name)
    else:
         plt.title(model_name+": "+embed_name[column])
    print(column, flush=True)
    legend_elements = []
    plt.xlabel("wrt. "+wrt_column+",n_neigh = "+str(options.n_neighbors))
    for label in options.labels:
        i = 0
        marker = "," #{"en":"x", "fr":".", "zh":"1", "ur": "+", "fi":"*", "tr": "_", "fa": "d", "sv":"3"}[lang]
        #legend_elements.append(Line2D([0],[0], color='w', markeredgecolor='black', markerfacecolor='black', marker=marker, label=lang, markersize=10))
        for lang, group in df[df[wrt_column]==label].groupby("lang"):
            if label==[] or label=="[]":
                continue
            plt.scatter(
                group["x_"+column],
                group["y_"+column],
                s=5,
                marker=marker,
                label=label,
                edgecolor="none",
                c=[color_palette[i]],#skipping some colors
                alpha=0.3,
            )
            if label == options.labels[0]: # legend only once; duplacates else.
                legend_elements.append(Line2D([0],[0], color='w', markerfacecolor=color_palette[i], marker="s", label=lang))
            i +=1

    lgnd = plt.legend(handles=legend_elements)
    for handle in lgnd.legend_handles:
        handle._sizes = [60]

    Path(options.save_path).mkdir(parents=True, exist_ok=True)
    
    fig_save_path = options.save_path+"langs_fig_"+embed_name[column]+"_"+wrt_column
    if options.seed != None:
        fig_save_path += "_seed_"+str(options.seed)
        
    plt.savefig(fig_save_path+".png")
    plt.close()
#fig.savefig("full.png")
