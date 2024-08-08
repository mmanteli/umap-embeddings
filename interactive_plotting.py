import pandas as pd
import numpy as np
import json
import umap #trimap
import plotly.express as px
import os
import sys
import re
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter



# NOTE:
# Text + embeds availabel for [bge, xlmr1, xlmr5] x [balanced reg oscar]
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
    ap.add_argument('--model_name', type=str, metavar='STR', default=None, required=True,
                    help='Name of the model for plot titles.')
    ap.add_argument('--data_name', type=str, metavar='STR', default=None,
                    help='If given, added to plot title.')
    ap.add_argument('--use_column', '--column', type=str, metavar='STR', default="preds_best", choices=["preds","labels", "preds_best"],
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


embed_name = {"embed_first":"first_layer","embed_half":"halfway_layer","embed_last":"last_layer"}



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
    print("len: ",len(df), flush=True)
    print(df.head(), flush=True)
    print(df.tail(), flush=True)
    print("label distribution: ", np.unique(df[wrt_column], return_counts=True))
    print("language distribution: ", np.unique(df["lang"], return_counts=True))
    df = df.explode(wrt_column)  # Does something and is needed here even if done before
    df = df.reset_index()
    return df

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






#--------------------------------Joonatan's code------------------------------#

def wrap_text(text, width, truncate=True):
    """Wrap text with a given width."""
    text_ = text[0:500]
    return '<br>'.join([text_[i:i+width] for i in range(0, len(text_), width)])


def plot_embeddings(df_plot, column, wrt, output_dir):
    
    df_plot["hover_text"] =  df_plot.apply(lambda row: f"{wrap_text(row['text'], 80)}", axis=1)

    fig = px.scatter(df_plot, x='x_'+column, y='y_'+column, color=wrt,
                     title='Embeddings',
                     #hover_data={"hover_text":True, "x_"+column:False, "y_"+column:False, "lang"=False},
                     hover_data={"lang":True, "preds_best":True, "hover_text":True, "text":False, "x_"+column:False, "y_"+column:False},
                     #hover_name='hover_text', hover_data={"x_"+column:True, "y_"+column:True,'lang': True, 'text': True},
                     width=2000, height=1500)  # Increased size for better visibility

    fig.update_layout(
        legend_title_text='Cluster',
        hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell", bordercolor="black"),
    )
    
    # Save the figure as an HTML file
    html_file = os.path.join(output_dir, f'{wrt}_{column}_kuva2.html')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
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

def main():
    """
    csv_file_path = 'All_Stories_embeddings.csv'
    output_dir = 'cluster_visualizations'
    n_clusters = 1500

    df, embeddings = load_and_prepare_data(csv_file_path)
    embeddings_reduced = reduce_dimensions(embeddings)
    cluster_labels = cluster_embeddings(embeddings_reduced, n_clusters)
    combined_text = df['combined_text'].tolist()  # Assuming you have this column
    plot_embeddings(embeddings_reduced, df['index'].tolist(), combined_text, cluster_labels, output_dir)
    """
    
    options = argparser().parse_args(sys.argv[1:])
    languages_as_list = options.languages.split(",")
    languages_as_list.sort()
    options.languages_as_list= languages_as_list
    options.labels.sort()
    options.remove_multilabel = bool(eval(options.remove_multilabel))
    if options.embeddings[-1] == "/":
        options.embeddings = options.embeddings[:-1]
    if options.save_path is None:
        lang_folder = "-".join(options.languages_as_list)
        label_folder = "" if options.labels==all_labels else "-large-labels" if options.labels==big_labels else "-"+"-".join(options.labels)
        options.save_path = options.embeddings.replace("model_embeds", "umap-figures") + "/"+lang_folder+label_folder+"/"
    print("-----------------INFO-------------------", flush=True)
    print(f'Loading from: {options.embeddings+"/"}', flush=True)
    print(f'Using languages {options.languages_as_list}')
    print(f'Saving to {options.save_path}', flush=True)
    print(f'Plotting based on column: {options.use_column}', flush=True)
    print("-----------------INFO-------------------", flush=True)
    # UMAP settings
    if options.seed is not None:
        reducer = umap.UMAP(random_state=options.seed, n_neighbors=options.n_neighbors)
    else:
        reducer = umap.UMAP(n_neighbors=options.n_neighbors)

    df = read_data(options)
    apply_umap(df, reducer)


    for column in embed_name.keys():
        plot_embeddings(df, column, options.use_column, "testi")
    for column in embed_name.keys():
        plot_embeddings(df, column, "lang", "testi")
        

if __name__ == "__main__":
    main()