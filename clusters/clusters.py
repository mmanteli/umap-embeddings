import numpy as np
import pandas as pd
import sys
import os
import re
from sklearn.metrics import silhouette_score, adjusted_rand_score, homogeneity_score, v_measure_score, completeness_score
from statistics import mean
#import matplotlib.pyplot as plt
import umap
import json
from sklearn.preprocessing import StandardScaler
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

all_labels = ["HI","ID","IN","IP","LY","MT","NA","OP","SP"]
big_labels = ["HI","ID","IN","IP","NA","OP"]

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
            print(f'Unable to sample to {n} > len(df) = {len(df)}.', flush=True)
        return df


def read_data(options):
    wrt_column = options.use_column
    dfs = []
    for filename in os.listdir(options.embeddings):
        file = os.path.join(options.embeddings, filename)
        #if any([l in filename.replace(".tsv","") for l in options.languages_as_list]):   # only take languages of intrest
        if filename.split("_")[0] in options.languages_as_list:
            print(f'Reading {file}...', flush=True)
            df = pd.read_csv(file, sep="\t")
            # rename the best f1 column from preds_{value} to preds_best
            df.rename(columns=lambda x: re.sub('_0.*','_best',x), inplace=True)
            # sample down to 2*sample to make running things faster
            df = do_sampling(df,options.sample, r=2)
            # from str to list, wrt_column contains the column we are interested in
            #print(df)
            #print(df.columns)
            try:
                df[wrt_column] = df[wrt_column].apply(
                    lambda x: eval(x)
                )
            except:   # CORE labels separated by white space, and NA maps to NaN in eval
                df[wrt_column] = df[wrt_column].apply(
                    lambda x: ["NA" if i=="nan" else str(i) for i in str(x).split(" ")]
                )
            # remove multilabel
            if options.remove_multilabel:
                df = df[df[wrt_column].apply(lambda x: len(x) < 2 and len(x) > 0)]
            else:
                # only remove empty, explode multilabel
                df = df.explode(wrt_column)
                df = df[df[wrt_column].apply(lambda x: len(x) > 0)]
            if options.labels != all_labels:     # some classes need to be removed
                    print(f'Removing all classes except {options.labels}', flush=True)
                    df = df[df[wrt_column].apply(lambda x: x not in options.labels)]  # remove unwanted classes
            # sample to options.sample
            #print(len(df))
            df = do_sampling(df,options.sample) 
            dfs.append(df)
    
    
    # concat and remove multilabel
    df = pd.concat(dfs)
    del dfs
    #print("len: ",len(df), flush=True)
    #print(df.head(), flush=True)
    #print(df.tail(), flush=True)
    print("label distribution: ", np.unique(df[wrt_column], return_counts=True))
    print("language distribution: ", np.unique(df["lang"], return_counts=True))
    df = df.explode(wrt_column)  # Does something and is needed here even if done before
    df = df.reset_index()
    return df


def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--embeddings', '--data', type=str, default=None, required=True, metavar='DIR',
                    help='Path to directory where precalculated embeddings are. Lang name assumed in the filename.')
    ap.add_argument('--languages','--langs','--language','--lang', type=str, metavar='LIST', default=None, required=True,
                    help='Which languages to download from --embeddings, give as: en,fr,zh.')
    ap.add_argument('--labels', type=json.loads, metavar='ARRAY-LIKE', default=all_labels,
                    help='Labels. Give as: \'["IN","NA"]\'. Others discarded. Works only with --remove_multilabel.')
    ap.add_argument('--sample', type=int, metavar='INT', default=2000,
                    help='How much to sample from each language.')
    ap.add_argument('--use_column', '--column', type=str, metavar='STR', default="preds_best", choices=["preds","labels", "preds_best"],
                    help='Which column containing labels to use, "preds"=th0.5, "labels"=TP, "preds_best"=preds with best th.')
    ap.add_argument('--remove_multilabel', default="True",
                    help='Remove docs with multilabel predictions.')
    ap.add_argument('--n_neighbors', type=int, metavar='INT', default=15,
                    help='How many neighbors for UMAP.')
    ap.add_argument('--seed', type=int, metavar='INT', default=None,
                    help='Seed for reproducible outputs. Default=None, UMAP runs faster with no seed.')
    ap.add_argument('--save_path', type=str, metavar='DIR', default=None,
                    help='Where to save results.')
    return ap


def apply_umap(df, reducer):
    # Values from string to list and flatten, get umap embeds
    for column in ["embed_first","embed_half","embed_last"]:
        df[column] = df[column].apply(
            lambda x: np.array([float(y) for y in eval(x)[0]])
        )
        scaled_embeddings = StandardScaler().fit_transform(df[column].tolist())
        red_embedding = reducer.fit_transform(scaled_embeddings)
        df["x_"+column] = red_embedding[:, 0]
        df["y_"+column] = red_embedding[:, 1]

def apply_von_mises(embeddings,n, labels):
    """
     Attributes
    |  ----------
    |
    |  cluster_centers_ : array, [n_clusters, n_features]
    |      Coordinates of cluster centers
    |
    |  labels_ :
    |      Labels of each point
    |
    |  inertia_ : float
    |      Sum of distances of samples to their closest cluster center.
    |
    |  weights_ : array, [n_clusters,]
    |      Weights of each cluster in vMF distribution (alpha).
    |
    |  concentrations_ : array [n_clusters,]
    |      Concentration parameter for each cluster (kappa).
    |      Larger values correspond to more concentrated clusters.
    |
    |  posterior_ : array, [n_clusters, n_examples]
    |      Each column corresponds to the posterio distribution for and example.
    |
    |      If posterior_type='hard' is used, there will only be one non-zero per
    |      column, its index corresponding to the example's cluster label.
    |
    |      If posterior_type='soft' is used, this matrix will be dense and the
    |      column values correspond to soft clustering weights.

    """
    vmf_soft = VonMisesFisherMixture(n_clusters=n, posterior_type='soft').fit_transform(embeddings) ###posterior_type='hard'
    #vmf_labels = vmf_soft.labels_
    print(vmf.concentrations_)
    print(vmf.score(embeddings, labels))


def main():
    options = argparser().parse_args(sys.argv[1:])
    languages_as_list = options.languages.split(",")
    languages_as_list.sort()
    options.languages_as_list= languages_as_list
    options.labels.sort()

    # read the data
    df = read_data(options)
    
    # UMAP settings
    if options.seed is not None:
        reducer = umap.UMAP(random_state=options.seed, n_neighbors=options.n_neighbors)
    else:
        reducer = umap.UMAP(n_neighbors=options.n_neighbors)
    # apply umap to embeddings
    apply_umap(df, reducer)

    embeddings = np.array(df[["x_embed_last", "y_embed_last"]])
    labels = np.array(df[["preds_best"]])

    print(embeddings[0:2])
    print(labels[0:10])

if __name__ == "__main__":
    main()