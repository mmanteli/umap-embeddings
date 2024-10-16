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
from spherecluster_spherical_kmeans import SphericalKMeans
from sklearn.decomposition import PCA
import umap
from sklearn.preprocessing import LabelEncoder
from jsonargparse import ArgumentParser
from jsonargparse import ActionYesNo
from jsonargparse.typing import Path_drw, Path_dc  # path that can be read, patch that can be created

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.express as px

# data reading
from read_embeddings import read_and_process_data

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
COLORSEPARATION = 0.33


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
ap.add_argument('--embeddings', '--data', type=Path_drw, required=True, metavar='DIR',
                help='Path to directory where precalculated embeddings are as csv/tsv. Language name separated by _ \
                    (e.g. emb_fr.tsv, sv_emb.tsv) assumed in filenames.')
ap.add_argument('--clustering_method','--cmethod', default="all", choices=["all","kmeans","spherical-kmeans"],
                help='Which clustering method to use, default=all.')
ap.add_argument('--reduction_method','--rmethod', default="pca", choices=["pca", "umap", "all"],
                help='Which dimension resuction to use, default=all.')
ap.add_argument('--no_reduction','--nr', default=False, type=bool,
                help='Option to use the full embedding data.')
ap.add_argument('--pca_before_umap', default=None, type=int,
                help='apply pca before umap, give as a interger dimension.')
ap.add_argument('--n_clusters', '--num_clusters', type=parse_range_number, metavar='[INT, INT, (INT step)]', default=[3,21],
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
ap.add_argument('--save_dir', '--output_dir', type=str, metavar='DIR', default="metric_plots/",
                help='Dir where to save the results to.')
ap.add_argument('--save_prefix', '--output_prefix', type=str, metavar='str', default="silh_and_ari",
                help='Prefix for save file, lang/label and other params added as well as file extension.')
ap.add_argument('--header', type=list[str], metavar='LIST[COLUMNS]', default=None,
                help='If the data has no column names, give them here.')




# ANOTHER POSSIBILITY FOR SPHERICAL KMEANS
"""
# from https://stackoverflow.com/questions/5529625/is-it-possible-to-specify-your-own-distance-function-using-scikit-learn-k-means
import sklearn.cluster._kmeans as kmeans
from sklearn.metrics import pairwise_distances

def custom_distances(X, Y=None, Y_norm_squared=None, squared=False):
    if squared: #squared equals False during cluster center estimation
        return pairwise_distances(X,Y, metric='minkowski', p=1.5)
    else:
        return pairwise_distances(X,Y, metric='minkowski', p=1.5)
    
kmeans._euclidean_distances = custom_distances
kmeans.euclidean_distances = custom_distances # utilized by the method `KMeans._transform`


km = kmeans.KMeans(init="k-means++", n_clusters=clusters, n_init=4, random_state=0)
"""

import sklearn.cluster._kmeans as sphe_kmeans
from sklearn.metrics import pairwise_distances

def cosine_distance(X, Y=None, Y_norm_squared=None, squared=False):
    if squared: #squared equals False during cluster center estimation
        return pairwise_distances(X,Y, metric='cosine')
    else:
        return pairwise_distances(X,Y, metric='cosine')

def euclidian_distance(X, Y=None, Y_norm_squared=None, squared=False):
    if squared: #squared equals False during cluster center estimation
        return pairwise_distances(X,Y, metric='euclidean')
    else:
        return pairwise_distances(X,Y, metric='euclidean')


def spherical_kmeans_sklearn_modified(x: np.array, n_clusters: int, n_iterations=300):
    sphe_kmeans._euclidean_distances = cosine_distance
    sphe_kmeans.euclidean_distances = cosine_distance # utilized by the method `KMeans._transform`
    km = sphe_kmeans.KMeans(n_clusters=n_clusters, max_iter=n_iterations)
    labels = km.fit_predict(x)
    return labels

def kmeans_sklearn_modified(x: np.array, n_clusters: int, n_iterations=300):
    sphe_kmeans._euclidean_distances = euclidian_distance
    sphe_kmeans.euclidean_distances = euclidian_distance # utilized by the method `KMeans._transform`
    km = sphe_kmeans.KMeans(n_clusters=n_clusters, max_iter=n_iterations)
    labels = km.fit_predict(x)
    return labels

def spherical_kmeans_nltk(x: np.array, n_clusters: int, n_iterations=300):
    
    # nltk had an options to define the metric; thus using this
    # however; this took ages to converge!!!!
    kmeans = KMeansClusterer(n_clusters, distance=cosine_distance, repeats=n_iterations)
    try:
        labels = kmeans.cluster(x, assign_clusters=True, trace=True)
    except:
        return None

    return labels

    
def regular_kmeans(x: np.array, n_clusters: int, n_iterations=300):
    """ Regular k-means clustering """
    
    kmeans = KMeans(n_clusters=n_clusters, max_iter=n_iterations, init="k-means++")
    labels = kmeans.fit_predict(x)

    #return kmeans.cluster_centers_, kmeans.labels_
    return labels


def calculate_silhouette_score(x, labels):
    """
    Calculate the silhouette score for the given DataFrame and cluster labels.
    """
    score = silhouette_score(x, labels)
    
    return score


def calculate_ari(true_labels, predicted_labels):
    """
    Calculate the Adjusted Rand Index (ARI) between two sets of labels.
    """
    # Calculate ARI
    ari = adjusted_rand_score(true_labels, predicted_labels)
    return ari


def apply_pca(x: np.array, n_dim: int, options):
    if options.seed is not None:
        pca = PCA(n_components=n_dim, random_state=options.seed)
    else:
        pca = PCA(n_components=n_dim)
    z = pca.fit_transform(x)
    return z

def apply_umap(x: np.array, n_dim: int, options):
    if options.seed is not None:
        reducer = umap.UMAP(n_neighbors=options.n_neighbors, min_dist=options.min_dist, n_components=n_dim, random_state=options.seed)
    else:
        reducer = umap.UMAP(n_neighbors=options.n_neighbors, min_dist=options.min_dist, n_components=n_dim)

    # if given in params, apply pca before umap:
    if options.pca_before_umap is not None:
        x = apply_pca(x, options.pca_before_umap, options)
    
    z = reducer.fit_transform(x)
    return z

def parse_to_float(df: pd.DataFrame, column):
    """
    Parses a pandas dataframe column(s) to list of floats, as it is most likely read as string.
    Give column name as string.
    Formats for column values can be
    "[[x1, x2, x3, ...]]" OR "x1 x2 x3 ..."
    Read comments to implement other formats.
    """
    for column in options.use_column_embeddings:
        try:
            df[column] = df[column].apply(
                lambda x: np.array([float(y) for y in eval(x)[0]])   # remove [0], if your embeds are [x1, x2, ...]
            )
        except:
            try:
                df[column] = df[column].apply(
                    lambda x: np.array([float(y) for y in np.fromstring(x, sep=" ")])   # change sep, if your embeds are "x1,x2,..."
                )
            except:
                print("Cannot change the data type of the embedding columns to float. Try to change the separator in the \
                      codeblock above, if your embeds are separated by something else than white space, or remove [0], if \
                      your data is not doubly nested.")

    
def cluster_loop(x, y, options):
    """
    apply selected clustering methods to x, calculate it's silhouette score
    and ARI wrt given labels y, for number of clusters given in options as
    options.n_clusters = [c1, c2, (interval)]
    Returns results as a dict with cluster method as the first key, 
    cluster number as the second key
    and silhouette + ARI as third key.
    """

    # redirect funtion based on --clustering_method
    cluster_ = lambda x: {"kmeans":kmeans_sklearn_modified, "spherical-kmeans":spherical_kmeans_sklearn_modified}[x]

    # result collection
    results = {k: {} for k in options.clustering_method}
    
    for method in options.clustering_method:
        print(f"In method {method}.")
        x_clus = []
        y_ari = []
        y_silh = []
        for d in range(*options.n_clusters):
            cluster_labels=cluster_(method)(x, d)    #redirects to correct function
            if cluster_labels is not None:
                silh = calculate_silhouette_score(x, np.array(cluster_labels).reshape(-1,))
                ARI = calculate_ari(y, np.array(cluster_labels))
                # append to results
                x_clus.append(d)
                y_silh.append(silh)
                y_ari.append(ARI)
            else:
                print(f"Problem with {method}.")
        results[method]["x"] = x_clus
        results[method]["y_silh"] = y_silh
        results[method]["y_ari"] = y_ari
    return results

   

def reduction_loop(x, y, options):
    # to be implemented; not only one pca for testing

    reduction_results = {}
    if options.n_pca != []:
        for d in range(*options.n_pca):
            print(f"In step pca={d}.")
            x_ = apply_pca(x, d, options)
            results = cluster_loop(x_, y, options)
            reduction_results["pca_"+str(d)] = results

    if options.n_umap != []:
        for d in range(*options.n_umap):
            print(f"In step umap={d}.")
            x_ = apply_umap(x, d, options)
            results = cluster_loop(x_, y, options)
            reduction_results["umap_"+str(d)] = results

    if options.no_reduction:
        print(f"In step no reduction.")
        results = cluster_loop(x, y, options)
        reduction_results["no_reduction"] = results
        
    return reduction_results

def create_vector(sectors,points,separation):
    len_empty_space = sectors*separation  # space (%) on the colorwheel to be left unused

    len_sector = (1-len_empty_space)/sectors    # how long each used sector is

    buffer = separation/2.0   # beginning apply only half of buffer
    result = []
    for s in range(0, sectors):
        p = np.linspace(buffer, buffer+len_sector, points)
        result.append([i for i in p])
        buffer += len_sector + separation
    return result


def create_colormap(results, methods):
    """
    creates a colormap that uses same hues for methods, but separates reduction methods.
    results = {"reduction method": {"cluster method": {...data...}}}
    methods = ["kmeans", "spherical-kmeans"]] etc.
    """
    reductions = [i for i in results.keys()]
    n_pca = len(reductions)   # lazy naming
    n_methods = len(methods)
    base_colormap_name="mrybm"
    separation = COLORSEPARATION/n_methods
    color_selection_vector = create_vector(n_methods, n_pca, separation)
    colors = {}
    assert len(color_selection_vector) == n_methods
    assert len(color_selection_vector[0]) == n_pca
    for sublist, m in zip(color_selection_vector, methods):
        c_map = px.colors.sample_colorscale(base_colormap_name,sublist)
        colors[m] = {}
        for c, p in zip(c_map, reductions):
            colors[m][p] = c
    return colors

def plot_results(results, options):

    num_labels=len(options.labels)

    # Initialize the figure
    # fig = go.Figure()
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Silhouette score", "ARI"))

    # make a colormap that has correct number of colors
    #num_methods = len(options.clustering_method)
    #num_reductions = len(results.keys())
    colormap = create_colormap(results, options.clustering_method)  

    for red_value, data in results.items():
        for method, values in data.items():
            x_vals = values['x']
            y1_vals = values['y_silh']
            y2_vals = values['y_ari']
            
            # Add trace for y1
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y1_vals,
                mode='lines',
                line=dict(color=colormap[method][red_value]),
                legendgroup=red_value,
                name=f'{red_value} {method}'
                ),
                row=1, col=1
            )
            
            # Add trace for y2 (same color but different dash)
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y2_vals,
                mode='lines',
                line=dict(color=colormap[method][red_value]),
                legendgroup=red_value,
                showlegend=False,
                name=f'{red_value} {method}'
                ),
                row=1, col=2
            )
            
    fig.add_vline(x=num_labels, line_dash="dot", line_color="gray", row="all", col="all", annotation_text="n_labels")
    num_langs = len(options.languages)
    if num_langs > 1:
        fig.add_vline(x=num_langs, line_dash="dot", line_color="gray", row="all", col="all", annotation_text="n_langs")
        if num_langs*num_labels <= options.n_clusters[1]: # top value of calculations
            fig.add_vline(x=num_langs*num_labels, line_dash="dot", line_color="gray", row="all", col="all", annotation_text="n_langs*n_langs")

    # Update layout
    fig.update_layout(
        title=f"Cluster metrics embeddings of {options.model_name} from {options.data_name} with {options.languages}.",
        #xaxis_title="number of clusters",
        #yaxis_title="scores",
        legend_title="Legend (click & double click)",
    )
    fig.update_xaxes(title_text="Number of clusters", row=1, col=1)
    fig.update_xaxes(title_text="Number of clusters", row=1, col=2)
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=1, col=2)
    langs = "-".join(options.languages)
    save_name = os.path.join(options.save_dir, options.save_prefix+"_"+langs+"_sample"+str(options.sample)+".html")
    pio.write_html(fig, file=save_name, auto_open=False)



def parse_params_further(options):
    """
    This function exists because I do not understand jsonargparse.
    """

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

    if options.no_reduction:
        if "spherical-kmeans" in options.clustering_method:
            print("Setting clustering method to k-means only, as no_reduction spherical is not implemented/is too slow.")
            options.clustering_method = ["kmeans"]
            
    return options

if __name__=="__main__":
    options = ap.parse_args(sys.argv[1:])
    # parse this mfs
    options = parse_params_further(options)
    
    print(options)
    
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
        results = reduction_loop(x, y, options)

        print("\nCalculations done, plotting...")
        plot_results(results, options)


    

