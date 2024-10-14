import pandas as pd
import numpy as np
import umap  # trimap, pacmap etc.
#import umap.plot
import plotly.express as px
import os
import sys
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from jsonargparse import ArgumentParser
from jsonargparse import ActionYesNo
from jsonargparse.typing import Path_drw, Path_dc  # path that can be read, patch that can be created

from read_embeddings import read_and_process_data

# The columns expected in the data by default
embedding_columns=["embed_first", "embed_half", "embed_last"]

# CORE scheme labels, and different combinations
all_labels = ["HI","ID","IN","IP","LY","MT","NA","OP","SP"]
main_labels = ["HI","ID","IN","IP","NA","OP"]
without_MT = ["HI","ID","IN","IP","LY","NA","OP","SP"]
sublabels = ["it", "os", "ne", "sr", "nb", "on", "re","oh", "en", "ra", "dtp", "fi", "lt", "oi", "rv","ob", "rs", "av", "oo""ds", "ed", "oe"]

labels_all_hierarchy_with_other = {
    "MT": ["MT"],
    "LY": ["LY"],
    "SP": ["SP", "it", "os"],
    "ID": ["ID"],
    "NA": ["NA", "ne", "sr", "nb", "on"],
    "HI": ["HI", "re", "oh"],
    "IN": ["IN", "en", "ra", "dtp", "fi", "lt", "oi"],
    "OP": ["OP", "rv", "ob", "rs", "av", "oo"],
    "IP": ["IP", "ds", "ed", "oe"],
}
# above dictionary reversed for easier lookup
reverse_hierarchy = {}
for main_label, sub_labels in labels_all_hierarchy_with_other.items():
    for sub_label in sub_labels:
        reverse_hierarchy[sub_label] = main_label


# this needed for CORE scheme, as NA is read as NaN
remove_nan = lambda x: "NA" if x == "nan" else str(x)

# this for saving figs:
fig_label = {"label_for_umap":"register", "lang":"language"}

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

def parse_save_directory(d):
    try:
        os.makedirs(d, exist_ok=True)
        return d
    except:
        print(f"Cannot create saving directory {d}, using default value (umap-figures/).")
        return "umap-figures/"
        

# ---------------------------------------------------ARGUMENTS--------------------------------------------------- #

ap = ArgumentParser(prog="plot_embeddings.py", description="Plot pre-calculated embeddings. Give \
                    input data as a pandas dataframe, define which columns to use and where to save figures.")
ap.add_argument('--embeddings', '--data', type=Path_drw, required=True, metavar='DIR',
                help='Path to directory where precalculated embeddings are as csv/tsv. Language name separated by _ \
                    (e.g. emb_fr.tsv, sv_emb.tsv) assumed in filenames.')
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
ap.add_argument('--pca','--n_pca', type=int, metavar='INT>0', default=None,
                help="Number of dimensions mapped to with PCA before UMAP. No value = No PCA.")
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
ap.add_argument('--save_dir', '--output_dir', type=parse_save_directory, metavar='DIR', default="umap-figures/",
                help='Dir where to save the results to.')
ap.add_argument('--save_prefix', '--output_prefix', type=str, metavar='str', default="umap_",
                help='Prefix for save file, lang/label and other params added as well as file extension.')
ap.add_argument('--header', type=list[str], metavar='LIST[COLUMNS]', default=None,
                help='If the data has no column names, give them here.')



#------------------------------------------------UMAP----------------------------------------------------#


def apply_reducer(df, reducer, options):
    # Values from string to list and flatten, get umap embeds
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
        scaled_embeddings = StandardScaler().fit_transform(df[column].tolist())
        if options.pca is not None:
            pca = PCA(n_components=options.pca)
            scaled_embeddings = pca.fit_transform(scaled_embeddings)
        red_embedding = reducer.fit_transform(scaled_embeddings)
        df["x_"+column] = red_embedding[:, 0]
        df["y_"+column] = red_embedding[:, 1]
    return df



#------------------------------------------------plotting-----------------------------------------------#

def plot_embeddings_normal(df_plot, data_column, color_column, options):

    title = f'Embeddings with {options.model_name} from {options.data_name}'

    print(f"Now plotting {'x_'+data_column},{'y_'+data_column} with coloring based on {color_column}.")
    
    fig = px.scatter(df_plot, x='x_'+data_column, y='y_'+data_column, color=color_column,
                     title=title, labels={"label_for_umap": "Register", "lang":"Language"},
                     width=1200, height=900)  # Increased size for better visibility

    
    fig.update_traces(marker={"opacity":0.5, "size":3})
    # freeze legend
    fig.update_layout(legend={'itemsizing': 'constant'})
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        })
    
    # Save the figure as an HTML file or png
    fig_file = os.path.join(options.save_dir, f'{options.save_prefix}_wrt_{fig_label[color_column]}_{data_column}.{options.extension}')
    if not os.path.exists(options.save_dir):
        os.makedirs(options.save_dir)
    if options.extension == "html":
        fig.write_html(fig_file)
    else:
        fig.write_image(fig_file)


def wrap_text(text, width, truncate=True):
    """Wrap text with a given width."""
    if truncate:
        text = text[0:500]
    return '<br>'.join([text[i:i+width] for i in range(0, len(text), width)])


def plot_embeddings_with_hover(df_plot, data_column, color_column, options):
    df_plot["hover_text"] =  df_plot.apply(lambda row: f"{wrap_text(row[options.hover_text], 80, truncate=options.truncate_hover)}", axis=1)

    print(f"Now plotting interactive plots for {'x_'+data_column},{'y_'+data_column} with coloring based on {color_column}.")

    fig = px.scatter(df_plot, x='x_'+data_column, y='y_'+data_column, color=color_column,
                     title=f'Embeddings with {options.model_name} from {options.data_name}',
                     labels={"label_for_umap": "Register", "lang":"Language"},
                     hover_data={"hover_text":True,"lang":True, "label_for_umap":True, "text":False, "x_"+data_column:False, "y_"+data_column:False},
                     width=1200, height=900)  # Increased size for better visibility
    fig.update_layout(legend= {'itemsizing': 'constant'})
    fig.update_traces(marker={"opacity":0.5, "size":3})
    #fig.update_layout(
    #    legend_title_text='Cluster',
    #    hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell", bordercolor="black"),
    #)
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        })
    # Save the figure as an HTML file
    html_file = os.path.join(options.save_dir, f'{options.save_prefix}_wrt_{fig_label[color_column]}_{data_column}.html')
    if not os.path.exists(options.save_dir):
        os.makedirs(options.save_dir)
    fig.write_html(html_file)
    
    # Add custom JavaScript for copying to clipboard
    with open(html_file, 'a') as f:
        f.write("""
<script>
    document.addEventListener('DOMContentLoaded', function() {
        var plot = document.querySelector('.plotly-graph-div');
        plot.on('plotly_click', function(data) {
            var infotext = data.points.map(function(d) {
                return d.customdata[0].replace(/<[^>]+>/g, '');  // Remove HTML tags for clean clipboard content
            });
            copyToClipboard(infotext.join('\\n\\n'));
        });
    });

    function copyToClipboard(text) {
        var el = document.createElement('textarea');
        el.value = text;
        document.body.appendChild(el);
        el.select();
        document.execCommand('copy');
        document.body.removeChild(el);
        var notification = document.createElement('div');
        notification.innerHTML = 'Copied to clipboard';
        notification.style.position = 'fixed';
        notification.style.bottom = '10px';
        notification.style.left = '10px';
        notification.style.padding = '10px';
        notification.style.backgroundColor = '#5cb85c';
        notification.style.color = 'white';
        notification.style.borderRadius = '5px';
        document.body.appendChild(notification);
        setTimeout(function() {
            document.body.removeChild(notification);
        }, 2000);
    }
</script>
        """)


if __name__=="__main__":
    options = ap.parse_args(sys.argv[1:])
    print(options)
    print("")
    df = read_and_process_data(options)
    if options.hover_text is not None:  # force this after hover_text is handld in read_and_process_data()
        options.extension = "html"
    if options.seed is not None:
        reducer = umap.UMAP(random_state=options.seed, n_neighbors=options.n_neighbors, min_dist=options.min_dist)
    else:
        reducer = umap.UMAP(n_neighbors=options.n_neighbors, min_dist=options.min_dist)
    # for a pacmap implementation?
    # reducer = pacmap.PaCMAP(n_neighbors=options.n_neighbors, apply_pca=True, MN_ratio=2, FP_ratio=1, random_state=seed)
    
    # apply reducer to embeddings, apply pca if stated in the options
    apply_reducer(df, reducer, options)
    
    # change which function to use based on hover (just rename the function to plot_embeddings)
    plot_embeddings = plot_embeddings_with_hover if options.hover_text is not None else plot_embeddings_normal

    # plot wrt labels
    for column in options.use_column_embeddings:
        color = "label_for_umap"
        plot_embeddings(df, column, color, options)
    # plot wrt language
    for column in options.use_column_embeddings:
        color = "lang"
        plot_embeddings(df, column, color, options)

    
    exit(0)