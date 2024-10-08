import numpy as np
import pandas as pd
import sys
import os
import re
from sklearn.metrics import silhouette_score, adjusted_rand_score, homogeneity_score, v_measure_score, completeness_score, f1_score
from sklearn.metrics import adjusted_rand_score
from statistics import mean
#import matplotlib.pyplot as plt
import umap
import json
from sklearn.preprocessing import StandardScaler
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from spherecluster import VonMisesFisherMixture
import matplotlib.pyplot as plt

all_labels = ["HI","ID","IN","IP","LY","MT","NA","OP","SP"]
big_labels = ["HI","ID","IN","IP","NA","OP"]


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





def apply_umap(df, reducer):
    # Values from string to list and flatten, get umap embeds
    for column in ["embed_last"]:
        df[column] = df[column].apply(
            lambda x: np.array([float(y) for y in eval(x)[0]])
        )
        scaled_embeddings = StandardScaler().fit_transform(df[column].tolist())
        red_embedding = reducer.fit_transform(scaled_embeddings)
        df["x_"+column] = red_embedding[:, 0]
        df["y_"+column] = red_embedding[:, 1]
        df["z_"+column] = red_embedding[:, 2]

def apply_von_mises(embeddings, labels, n):
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
    transform(self, X, y=None)
    |      Transform X to a cluster-distance space.
    |      In the new space, each dimension is the cosine distance to the cluster
    |      centers.  Note that even if X is sparse, the array returned by
    |      `transform` will typically be dense.
    |
    |      Parameters
    |      ----------
    |      X : {array-like, sparse matrix}, shape = [n_samples, n_features]
    |          New data to transform.
    |
    |      Returns
    |      -------
    |      X_new : array, shape [n_samples, k]
    |          X transformed in the new space.

    """
    vmf_soft = VonMisesFisherMixture(n_clusters=n, posterior_type='soft', verbose=True, normalize=True, random_state=123).fit(embeddings,labels) ###posterior_type='hard'
    #vmf_labels = vmf_soft.labels_
    clustered_embeds = vmf_soft.transform(embeddings)
    clustered_labels = vmf_soft.labels_

    print(clustered_labels[0:10])
    print(labels[0:10])
    print("Kappa value for clustering (larger value better)")
    print(vmf_soft.concentrations_)
    print("ARI?")
    print(adjusted_rand_score(labels, clustered_labels))
    #print("score wrt true labels (larger value better)")
    #print(vmf_soft.score(embeddings, labels))

    #vmf_soft = VonMisesFisherMixture(n_clusters=2, posterior_type='soft', n_init=20)
    #vmf_soft.fit(X)
    exit()
    print(clustered_embeds.shape)
    
    exit()
    fig = plt.figure(figsize=(8, 6))

    ax = fig.add_subplot(1, 1, 5, aspect='equal', projection='3d',
        adjustable='box-forced', xlim=[-1.1, 1.1], ylim=[-1.1, 1.1],
        zlim=[-1.1, 1.1])
    ax.scatter(X[vmf_soft.labels_ == vmf_soft_mu_0_idx, 0], X[vmf_soft.labels_ == vmf_soft_mu_0_idx, 1], X[vmf_soft.labels_ == vmf_soft_mu_0_idx, 2], c='r')
    ax.scatter(X[vmf_soft.labels_ == vmf_soft_mu_1_idx, 0], X[vmf_soft.labels_ == vmf_soft_mu_1_idx, 1], X[vmf_soft.labels_ == vmf_soft_mu_1_idx, 2], c='b')
    ax.set_aspect('equal')
    plt.title('soft-movMF clustering')
    plt.savefig("testi.png")

    # vmf_soft.cluster_centers_
    # vmf_soft.labels_
    # vmf_soft.weights_
    # vmf_soft.concentrations_
    # vmf_soft.inertia_


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
        reducer = umap.UMAP(random_state=options.seed, n_neighbors=options.n_neighbors, n_components=3)
    else:
        reducer = umap.UMAP(n_neighbors=options.n_neighbors,n_components=3)
    # apply umap to embeddings
    apply_umap(df, reducer)

    embeddings = np.random.rand(options.sample*2,3) #np.array(df[["x_embed_last", "y_embed_last", "z_embed_last"]])
    labels = np.array(df[["preds_best"]])

    print(embeddings.shape)
    print(labels[0:10])
    unique_labels = np.unique(labels)
    print(f'num labels in this sample = {len(unique_labels)}')

    label2id = {k:v for k,v in zip(unique_labels, range(len(unique_labels)))}
    print(label2id)
    labels_num = [label2id[i[0]] for i in labels]
    print(labels_num[0:10])
    apply_von_mises(embeddings, labels_num, len(np.unique(labels)))

if __name__ == "__main__":
    main()