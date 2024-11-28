"""
Functions for reading different types of embedding data.
Not to be used alone, these are imported into multiple files.
"""

import pandas as pd
import numpy as np
import os
import sys
import re

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
                #df = df.sample(n=options.sample)#, weights='label_for_umap', random_state=1)#.reset_index(drop=True)
                how_many = int(options.sample/len(options.labels))
                print(f"trying to sample {how_many} from each label with available: {df.value_counts('label_for_umap')}.")
                #df = df.sample(n=options.sample)#, weights='label_for_umap', random_state=1)#.reset_index(drop=True)
                df = df.groupby('label_for_umap').apply(lambda x: x.sample(n=how_many, replace=True, random_state=0)).reset_index(drop=True)
            
            # finally, drop everything unneeded and append
            if options.hover_text is not None:
                if all(i in df.columns for i in [options.hover_text]):
                    df = df[['lang','label_for_umap', *options.use_column_embeddings, options.hover_text]]
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