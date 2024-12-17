# umap-embeddings on LUMI
Dump for umap visualisations of documents embeddings and calculating metrics.

## What is this about?

*Hypothesis*:
Registers contain cross-lingual features and we hope to show that documents embedded with register classifier cluster wrt. register, and not by language. This means that the embdding space is organized wrt. linguistic features and not by language features. 

We train multiple models with multilingual manual register annotations and get the embeddings (last layer output before classification head) and associated labels for unseen documents. We then use k-means and spherical-kmeans to cluster these embeddings and evaluate them against the predicted (assumed true) labels.

The results show that both clustering methods provide best results when the number of clusters is close to number of registers in the data. This means that the natural clustering by register provides better explanation of embedding space behaviour than clustering by language.

## Getting started

This branch is LUMI specific and has had the most active development. 

### ``requirements.txt``

Embedding and predictions run smootly on LUMI-module ``PyTorch``.
The packages needed **for clustering** are listed in requirements. Specifically important is ``umap-learn``.

#### for embeddings
PyTorch module activation is done in embedding scripts automatically like this:
```
module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.4   # specifically 2.4 used here, for later reference
```
Now you can call ``python`` which redirect to ``pytorch/2.4``'s ``python3.10`` (or at least not ``python3.6``, which is default on LUMI).

#### for clustering
For the clustering, you can create a virtual environment with
```
module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.4
python -m venv .venv   # or any other name you want
source .venv/bin/activate
pip install -i requirements.txt  # -i == install things listed in requirements.txt
```
and this will be again activated in clustering scripts with ``source .venv/bin/activate``.

#### Note:
Using venv on LUMI is not recommended. It works here (at least for me), but it does not work with GPU partitions.
For that, you can create a ``PYTHONUSERBASE``, which works like a venv, but isn't one. See [here](https://github.com/mmanteli/register-model-training?tab=readme-ov-file#evaluation) for instructions. Adding the ``--user--`` flag changes the installation to variable ``$PYTHONUSERBASE``.


### Embeddings

You can calculate the embeddings and register predictions with ``run_embeds.sh`` which launches ``embeds.py``. Both assume quite a lot about the data structure, so change them accordingly. Parametres that ``embeds.py`` takes are:

- ``--model_name``: key in a dictionary inside ``embeds.py`` that points to the model's save location. Change your model paths there (and also modify accepted parameter values).
- ``--fold``: For different models with the same name, this is used to specify location. Similarly, see the model path dictionary. E.g. ``--model_name=bge-m3 --fold=1`` finds bg3 model number 1.  
- ``--data_name``: Similar to model name, finds the data location in a dictionary. Modify the dictionary for your own needs.
- ``--language``: Similar to ``--fold`` but for data, i.e. ``--data_name=CORE --language="en"`` finds CORE-en dataset.
- ``--f1_limits``: which values to search when trying to find best f1-score. Defaults to ``range(0.3, 0.65, 5) = [0.3, 0.35,...,0.6]``. These values are sufficient in almost all cases.
- ``return_text``: Whether to return text with the embeddings and labels. Handy if your planning to plot html plots where you can see the text interactively. Defaults to 1 which means yes. Set to 0 if you do not want text.
- ``seed``: seed for random behaviour.
- ``save_path``: you can define this to what ever you want, else ``/scratch/project_462000353/amanda/register-clustering/data/model_embeds/{options.data_name}/{options.model_name}-fold-{options.fold}/{options.language}.tsv`` is used. Change this preferably.

After running this, you have a .tsv file with embeddings from 3 different layers of the model, true labels and predicted labels for th=0.5 and th=best_f1 (column_name=``preds_<numerical value>``), and text, if you didn't set ``--return_text`` to 0.

### Plotting the embeddings

If you want to do some sanity checks before clustering, you can plot the embeddings with ``run_plotting.sh`` which launches ``plot_embeddings.py``. This plots the clusters with different colors for language and label. You can set the label to any column in the embedding tsv, with "preds_best" meaning column with best f1-threshold.

Dimensionality reduction is done using UMAP. You can also apply pca before UMAP, this was implemented to replecate earlier results when using UMAP on all +700 dimensions wasn't possible. This script only accepts 2D UMAP, as it plots, but ``clusters.py`` and ``clusters_mean.py`` do UMAP for other dimensions as well.

``plot_embeddings.py`` takes a lot of parameters, just the most important here:

- ``--embeddings``: path to the directory of embedding result tsv
- ``--languages``: which languages to read from embeddings path, they are assumed to be saved as en.tsv etc.
- ``--use_column_labels``: which column to read the labels from. As described above, "preds_best" reads column ``preds_<numerical value>``, otherwise you can make it read any column name as it is.
- ``--use_column_embeddings``: similar to above, which column to find the embedding values from. You can give multiple as as list, default to all.
- ``--hover_text``: give the column that contains the texts (probably "text" if you didn't change anything) to have the text hover in an interactive plot. You can also put other columns here, if you want.
- ``--model_name`` and ``data_name``: these are for plot titles only.
- ``save_dir`` and ``--save_prefix``: directory where to save and prefix for the file name. Other stuff appended in the end to distinguish languages and label columns.

#### Note:
I have made minor modification to ``plot_embeddings.py`` after running ``run_plotting.sh``, so some minor bugs may have been created.

### Clustering

Clustering takes the embeddings, applies UMAP and then uses clustering algorithms to find different possible clusterings and then output a graph that tells you which one of them was the best. Clusters are evaluated using Silhouette score and Adjusted Random Index (ARI), pretty standard metrics.

For clustering, there are two options. You can do it on one model only and also plot the results similarly to above using ``sl-cluster-metrics.sh`` which launches ``clusters.py``. This creates cluster metric graph, and embedding plots for different scenarios:

- coloring by language
- coloring by true label
- coloring by best clustering for the two metrics, and for each specified clustering method (kmeans and/or spherical-kmeans)

If you want to accound for randomness, use ``sl-cluster-metrics-average.sh``, which launches ``clusters_mean.py``. This does basically what ``clusters.py`` does, however it averages the metrics over all the data it gets. Thus, no cluster plots. 

``clusters.py`` uses quite a bit of code from ``plot_embeddings.py`` and thus modifying it has consequences. Similarly ``clusters_mean.py`` uses code from both ``clusters.py`` and ``plot_embeddings.py``.

These take similar argument to ``plot_embeddings.py``. Additionally, you can define which UMAP dimensions to use (not limited to 2D), and of course set clustering method to kmeans or spherical-kmeans or both=``all``. **EXCEPT do not** use ``--clustering_method=all`` with ``cluster_means.py``, do them separately if needed. You can also correct this bug if you want, I think somewhere the dictionary keys overwrite each other.







