import numpy as np
import pandas as pd
import sys
import os
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from nltk.cluster.kmeans import KMeansClusterer
from nltk.cluster.util import cosine_distance
#from spherecluster_spherical_kmeans import SphericalKMeans
from sklearn.decomposition import PCA
import umap
from sklearn.preprocessing import LabelEncoder
from jsonargparse import ArgumentParser
from jsonargparse import ActionYesNo
from jsonargparse.typing import Path_drw, Path_dc  # path that can be read, patch that can be created
import glob
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.express as px
import pickle

# data reading
from read_embeddings import read_and_process_data
# data plotting
from plot_embeddings import plot_embeddings_normal, plot_embeddings_with_hover
# actual calculations
from clusters import parse_to_float, reduction_loop, plot_results

# The columns expected in the data by default
embedding_columns=["embed_last"]#, "embed_first" ,"embed_half", "embed_last"]

# CORE scheme labels, and different combinations
all_labels = ["HI","ID","IN","IP","LY","MT","NA","OP","SP"]
main_labels = ["HI","ID","IN","IP","NA","OP"]
without_MT = ["HI","ID","IN","IP","LY","NA","OP","SP"]
sublabels = ["it", "os", "ne", "sr", "nb", "on", "re","oh", "en", "ra", "dtp", "fi", "lt", "oi", "rv","ob", "rs", "av", "oo""ds", "ed", "oe"]

# percentage of the colorwheel not used in the plots;
# high values choose more similar colors for a method, e.g. blueish for kmeans, redish for spherical
# low values choose uniformly from colorwheel
COLORSEPARATION = 0.4


# This for easy parsing of label parameters; if given as a list, that is used, else checking for keywords
def parse_labels(l):
    if type(l)==list:
        return l
    elif l == "upper" or l == "all":
        return all_labels
    elif l == "without_MT":
        return without_MT
    elif l == "main":
        return main_labels
    else:
        print(f"ERRONIOUS LABELS; USING {main_labels}")


def parse_range_number(n):
    if type(n)==str and "[" not in n and "," in n:
        c = [int(i) for i in n.split(",")]
    elif type(n) == list:
        c = n
    elif type(eval(n)) == list:
        c = eval(n)
    else:
        print("Cluster/PCA number given incorrectly.")
        exit(1)
        
    assert len(c) == 2 or len(c) == 3   # two or three numbers needed for range()
    c[1] += 1    # to make the last value also accounted
    if len(c) == 2:
        c.append(1)   # step interval assumed 1 if nog given
    return c

        
ap = ArgumentParser(prog="clusters.py", description="Cluster pre-calculated embeddings.")
ap.add_argument('--embeddings', '--data', type=str, required=True, metavar='DIR',
                help='Path to directory where to-be-averaged embeddings are. Give as data/xlmr-*/ to read all xlmr files.')
ap.add_argument('--clustering_method','--cmethod', default="all", choices=["all","kmeans","spherical-kmeans"],
                help='Which clustering method to use, default=all.')
ap.add_argument('--reduction_method','--rmethod', default="umap", choices=["pca", "umap", "all"],
                help='Which dimension resuction to use, default=all.')
ap.add_argument('--no_reduction','--nr', default=False, type=bool,
                help='Option to use the full embedding data.')
ap.add_argument('--pca_before_umap', default=None, type=int,
                help='apply pca before umap, give as a interger dimension.')
ap.add_argument('--n_clusters', '--num_clusters', type=parse_range_number, metavar='[INT, INT, (INT step)]', default=None,
                help="number of clusters to be looped over. Give as [1,2] or [3,7,2]")
ap.add_argument('--languages','--langs','--language','--lang', type=list[str], metavar='LIST', required=True,
                help='Which languages to download from --embeddings path.')
ap.add_argument('--labels', type=parse_labels, metavar='LIST or GROUP NAME', default=main_labels,
                help='Labels as list )["IN", "NA"] etc.) or a group name. Others discarded. \
                      Group names = ["all", "main", "without_MT"].')
ap.add_argument('--use_column_labels', '--column_l', type=str, default="preds_best", metavar='COLUMN',
                help='Column name containing labels that are used for coloring the figure. \
                    If "preds_best" (default), assumes data contains column named preds_{threshold}.')
ap.add_argument('--use_column_embeddings', '--column_e', type=list, default=embedding_columns, metavar='LIST[COLUMN]',
                help='column(s) that contain embeddings. If multiple, separate figs are produced.')
ap.add_argument('--remove_hybrids', type=bool, metavar='BOOL', default=True,
                help=f'Remove docs with multiple main level ({all_labels}) predictions.')
ap.add_argument('--keep_sublabels', type=bool, metavar='BOOL', default=False,
                help='Keep sublabels and attach them to main level with hyphen, e.g. "HI-re"')
ap.add_argument('--n_pca','--pca', type=parse_range_number, metavar='[INT, INT]', default=[3,7,1],
                help="Number of dimensions mapped to with PCA as lower and higher thresholds.")
ap.add_argument('--n_umap','--umap', type=parse_range_number, metavar='[INT, INT]', default=[2,3,1],
                help="Number of dimensions mapped to with PCA as lower and higher thresholds.")
ap.add_argument('--plot_best_ARI', type=bool, default=False,
                help="Uses the plotting script to plot the best clustering wrt. ARI")
ap.add_argument('--hover_text', type=str, metavar="COLUMN", default=None, 
                help="Adds column values as hover text")
ap.add_argument('--truncate_hover', default=True,  type=bool, metavar="bool",
                help='Truncate hover text.')
ap.add_argument('--n_neighbors', type=int, metavar='INT', default=50,
                help='How many neighbors for UMAP.')
ap.add_argument('--min_dist', type=float, metavar='FLOAT', default=0.0,
                help='Minimum distance for UMAP.')
ap.add_argument('--sample', type=int, metavar='INT', default=None,
                help='How much to sample from each language, if given.')
ap.add_argument('--model_name', type=str, metavar="STR",
                help='Added to plot titles if given.')
ap.add_argument('--data_name', type=str, metavar="STR",
                help='Added to plot titles if given.')
ap.add_argument('--seed', type=int, metavar='INT', default=None,
                help='Seed for reproducible outputs. Default=None, UMAP runs faster with no seed.')
ap.add_argument('--extension', type=str, metavar="str", default="png", choices=["png", "html"],
                help="Which format for saving the plots. --hover_text forces html.")
ap.add_argument('--save_dir', '--output_dir', type=str, metavar='DIR',required=True,
                help='Dir where to save the results to.')
ap.add_argument('--save_prefix', '--output_prefix', type=str, metavar='str', default="silh_and_ari",
                help='Prefix for save file, lang/label and other params added as well as file extension.')
ap.add_argument('--header', type=list[str], metavar='LIST[COLUMNS]', default=None,
                help='If the data has no column names, give them here.')




def parse_params_further(options):
    """
    This function exists because I do not understand jsonargparse.
    """

    # number of clusters needed:
    if options.n_clusters is None:
        options.n_clusters = [2, len(options.labels)*len(options.languages)+1]

    # change this to something that can be looped through
    if options.clustering_method == "all":
        options.clustering_method = ["kmeans", "spherical-kmeans"]
    else:
        options.clustering_method = [options.clustering_method]
            
    try:
        os.makedirs(options.save_dir, exist_ok=True)
    except Exception as e:
        print("Cannot create save directory.")
        print(e)
        
    # make umap and pca ranges empty if needed:
    # this way, we don't need to write conditions, just loop regularly and not do anything
    if options.reduction_method == "pca":
        options.n_umap = []   # not looping through this
    if options.reduction_method == "umap":
        options.n_pca = []

    #if options.no_reduction:
    #    if "spherical-kmeans" in options.clustering_method:
    #        print("Setting clustering method to k-means only, as no_reduction spherical is too slow.")
    #        options.clustering_method = ["kmeans"]
            
    return options



def read_multiple_files(path_with_wildcards):
    if path_with_wildcards[-1] != "/":
        path_with_wildcards += "/"
    print(f"Reading from {path_with_wildcards}")
    # find all matching files
    tsv_files = glob.glob(path_with_wildcards)

    accessible_files = [t for t in tsv_files if os.path.exists(t)]
    if len(accessible_files)<len(tsv_files):
        print(f"Some files were not accessible (not readable/doenst exist).\n all files = {tsv_files}\n acc. files = {accessible_files}")
    
    return accessible_files
    

def the_loop_itself(options):
    print("\nReading data")
    df = read_and_process_data(options)
    print(f"\nParsing columns {options.use_column_embeddings} to float.")
    parse_to_float(df, options.use_column_embeddings)

    # the loop itself
    for column in options.use_column_embeddings:
        x = np.array(df[column].values.tolist())
        given_labels = np.array(df["label_for_umap"].values.tolist())

        print("\nNormalizing and encoding labels")
        # handle a couple things more (labels need to be numerical for ARI)
        x = normalize(x)
        Enc = LabelEncoder()
        y = Enc.fit_transform(given_labels)

        print("\nStarting pca + num-clusters loop...")
        results, data = reduction_loop(x, y, options)
    return results


def average_results(results):

    print("Trying to average....")

    mean_results = {}
    rmethods = results[0].keys()
    cmethods = results[0][list(rmethods)[0]].keys() # options.clustering_method
    for r in rmethods:
        for c in cmethods:
            mean_results[r+"_mean"] = {}
            mean_results[r+"_mean"][c] = {}
            x = []
            silh = []
            ari = []
            for d in results:
                x.append(d[r][c]["x"])
                silh.append(d[r][c]["y_silh"])
                ari.append(d[r][c]["y_ari"])
            mean_results[r+"_mean"][c]["x"] = np.mean(x,axis=0)
            mean_results[r+"_mean"][c]["y_silh"] = np.mean(silh,axis=0)
            mean_results[r+"_mean"][c]["y_silh_std"] = np.mean(silh,axis=0)
            mean_results[r+"_mean"][c]["y_ari"] = np.mean(ari, axis=0)
            mean_results[r+"_mean"][c]["y_ari_std"] = np.std(ari, axis=0)

    return mean_results
            
            
        

if __name__=="__main__":
    options = ap.parse_args(sys.argv[1:])
    # parse this mfs
    options = parse_params_further(options)
    print(options)
    
    files = read_multiple_files(options.embeddings)
    collect_results = []
    for f in files:
        options.embeddings = f
        results = the_loop_itself(options)
        collect_results.append(results)
    #print(collect_results)
    mean = average_results(collect_results)

    #print(mean)

    # Writing the data down for future reference, e.g. wanting to plot with or without error bars:
    p = os.path.join(options.save_dir, options.save_prefix+"_sample_"+str(options.sample)+"_plot_data.pkl")
    with open(p, 'wb') as f:
        pickle.dump(mean, f)

    
    print("\nCalculations done, plotting...")
    plot_results(mean, options)

    


    

