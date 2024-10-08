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
ap.add_argument('--hover_text', type=list[str], metavar="COLUMN(s)", default=None, 
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
ap.add_argument('--save_dir', type=Path_dc, metavar='DIR', default="umap-figures",
                help='Dir where to save the results to.')
ap.add_argument('--save_prefix', type=str, metavar='str', default="umap_",
                help='Prefix for save file, lang/label and other params added as well as file extension.')
ap.add_argument('--header', type=list[str], metavar='LIST[COLUMNS]', default=None,
                help='If the data has no column names, give them here.')


# --------------------------------------------------DATA READING-------------------------------------------------- #
def read_file(file, extension, names=None):
    if extension == ".tsv":
        if names is None:
            return pd.read_csv(file, sep="\t")
        else:
            return pd.read_csv(file, sep="\t", header=None, names=names)
    elif extension == ".csv":
        if names is None:
            return pd.read_csv(file, sep=",")
        else:
            return pd.read_csv(file, sep=",", header=None, names=names)
    elif extension in [".jsonl", ".json"]:
        if names is None:
            return pd.read_json(file)
        else:
            print("Column names not implemented for json as they are required in the format. Exiting.")
            exit(1)
    else:
        print("Extension must be .csv, .tsv, or json(l).")
        exit(1)


def separate_sub_labels_from_incorrect_main_labels(df):
    """
        This function is made for post-processing pandas.DataFrame.explode(). For example:
        1  NA ne OP
        2  LY
        ...
    
        explode(w.r.t main level label)
        => 
        1  NA ne
        1  OP ne
        2  LY
        ...
    
        This function removes the erroneous OP ne combination, but keeps the correct NA ne combination.
        1  NA ne
        1  OP
        2  LY
    """
    new_sublabels = []
    for index, d in df.iterrows():
        if type(d["subregister_prediction"]) == str:
            if d["subregister_prediction"] not in labels_all_hierarchy_with_other[d["register_prediction"]]:
                new_sublabels.append(np.nan)   # remove erronious => makes LY ID (and MT) be dropped!
                continue
                #e.g. NA + ob can be an error OR from NA, OP, ob
                if reverse_hierarchy[d["subregister_prediction"]] in d["full_register"]:
                    new_sublabels.append(np.nan)
                else:
                    new_sublabels.append(d["subregister_prediction"])
            else:
                new_sublabels.append(d["subregister_prediction"])
        else:
                new_sublabels.append(d["subregister_prediction"])
    df["subregister_prediction"] = new_sublabels
    return df

def read_and_process_data(options):
    """ 
        Function for reading embedding files and their associated languages and labels.
        Long, as there are many label combination options in the parameters.
        Reads data and creates new column "label_for_umap" that is used in plotting.
    """
    wrt_column = options.use_column_labels   # which column to read
    dfs = []
    for filename in os.listdir(options.embeddings):
        file = os.path.join(options.embeddings, filename)   # this is used for download
        _, file_extension = os.path.splitext(file)  # extension is used for download and selection
        # select files that have the given language(s) separated by _
        matched_language = (next((lang for lang in options.languages if lang in filename.replace(file_extension,"").split("_")),None))
        if matched_language:
            print(f'Reading {file}...', flush=True)
            df = read_file(file, file_extension, names=options.header)
            # Notify about language column missing:
            if 'lang' not in df.columns:
                df["lang"] = matched_language
                print(f"Language ({matched_language}) added automatically since column 'lang' was missing in the data.")
            if wrt_column == "preds_best":
                # rename the best f1 column from preds_{f1 threshold value} to preds_best
                df.rename(columns=lambda x: re.sub('_0.*','_best',x), inplace=True)
            assert wrt_column in df.columns, f"Given --use_column_labels={wrt_column} not in given data columns. Columns are {df.columns}"

            # start parsing labels
            try:
                df["full_register"] = df[wrt_column].apply(lambda x: eval(x))
            except Exception as e1: 
                print(f'Cannot evaluate given column {wrt_column}, trying with split(" ").')
                try:
                    df["full_register"] = df[wrt_column].apply(lambda x: [remove_nan(i) for i in str(x).split(" ")])   # "NA" label read as nan
                except Exception as e2:
                    print(f"Cannot read the given label column {wrt_column}. Both errors below:")
                    print(e1)
                    print(e2)
                    exit(1)

            # make separate column for next steps
            df["register_prediction"] = df["full_register"].apply(lambda x: [i for i in x if i in options.labels])

            # two possible options: remove hybrids (>1 main labels) or keep sublabels (make HI,re to a new HI-re category)
            if options.remove_hybrids:
                # removes all files with multiple main-level predictions
                # SO: if options.remove_hybrids + options.keep_sublabels, this removes "NA HI re" but not "HI re"
                df = df[df["register_prediction"].apply(lambda x: len(x) < 2 and len(x) > 0)]
            if options.keep_sublabels:
                # separate label levels further:
                df["register_prediction"] = df["full_register"].apply(lambda x: [i for i in x if i in options.labels])
                df["subregister_prediction"] = df["full_register"].apply(lambda x:  [i for i in x if i in sublabels])
                # explode wrt. both
                # This causes NA OP ob to be divided into two: NA+ob and OP+ob
                # correct this with separate_sub_labels_from_incorrect_main_labels()
                df = df.explode("register_prediction").explode("subregister_prediction")
                df = separate_sub_labels_from_incorrect_main_labels(df)
                
                # make a combined label, i.e. HI, re => HI-re
                df['label_for_umap'] = df.apply(
                    lambda row: f"{row['register_prediction']}-{row['subregister_prediction']}" if pd.notna(row['register_prediction']) and pd.notna(row['subregister_prediction'])
                    else row['register_prediction'] if pd.notna(row['register_prediction'])
                    else row['subregister_prediction'] if pd.notna(row['subregister_prediction'])
                    else np.nan,
                    axis=1
                    )
            else:
                df["label_for_umap"] = df["register_prediction"]   # just keep main level labels

            # If there are actual NaN's:
            df = df[df['label_for_umap'].notna()]

            # explode == flatten and separate hybrids etc. depending on remove_hybrids etc.
            df = df.explode("label_for_umap")

            # if we need to downsample:
            if options.sample:
                df = df.sample(n=options.sample)#, weights='label_for_umap', random_state=1)#.reset_index(drop=True)
            
            # finally, drop everything unneeded and append
            if options.hover_text is not None:
                if all(i in df.columns for i in options.hover_text):
                    df = df[['lang','label_for_umap', *options.use_column_embeddings, *options.hover_text]]
                else:
                    df = df[['lang','label_for_umap', *options.use_column_embeddings]]
                    print("--hover_text= {options.hover_text} given but columns could not be found. Setting as null.")
                    options.hover_text = None
            else:
                df = df[['lang','label_for_umap', *options.use_column_embeddings]]
            dfs.append(df)

    # after reading a processing, concatenate
    df = pd.concat(dfs)
    del dfs
    print(df.head(10))
    print("label distribution: ", np.unique(df["label_for_umap"], return_counts=True))
    print("language distribution: ", np.unique(df["lang"], return_counts=True))
    return df

#------------------------------------------------UMAP----------------------------------------------------#


def apply_umap(df, reducer, options):
    # Values from string to list and flatten, get umap embeds
    for column in options.use_column_embeddings:
        df[column] = df[column].apply(
            lambda x: np.array([float(y) for y in eval(x)[0]])
        )
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

    #fig.update_layout(
    #    legend_title_text='Cluster',
    #)
    
    # Save the figure as an HTML file or png
    fig_file = os.path.join(options.save_dir, f'{options.save_prefix}_wrt_{color_column}_{data_column}.{options.extension}')
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


def plot_embeddings_with_hover(df_plot, column, color, options):
    df_plot["hover_text"] =  df_plot.apply(lambda row: f"{wrap_text(row['text'], 80, truncate=options.truncate_text)}", axis=1)

    fig = px.scatter(df_plot, x='x_'+column, y='y_'+column, color=color,
                     title=f'Embeddings with {options.model_name} from {options.data_name}',
                     labels={"label_for_umap": "Register", "lang":"Language"},
                     #hover_data={"hover_text":True, "x_"+column:False, "y_"+column:False, "lang"=False},
                     hover_data={"hover_text":True,"lang":True, options.use_column:True, "text":False, "x_"+column:False, "y_"+column:False},
                     #hover_name='hover_text', hover_data={"x_"+column:True, "y_"+column:True,'lang': True, 'text': True},
                     width=2000, height=1500)  # Increased size for better visibility

    fig.update_layout(
        legend_title_text='Cluster',
        hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell", bordercolor="black"),
    )
    
    # Save the figure as an HTML file
    html_file = os.path.join(options.save_dir, f'{options.save_prefix}{color}_wrt_{options.use_column}_{column}.html')
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
    # apply umap to embeddings, apply pca if needed
    apply_umap(df, reducer, options)
    
    # change which function to use based on hover (just rename the function to plot_embeddings)
    plot_embeddings = plot_embeddings_with_hover if options.hover_text else plot_embeddings_normal

    # plot wrt labels
    for column in options.use_column_embeddings:
        color = "label_for_umap"
        plot_embeddings(df, column, color, options)
    # plot wrt language
    for column in options.use_column_embeddings:
        color = "lang"
        plot_embeddings(df, column, color, options)

    
    exit(0)