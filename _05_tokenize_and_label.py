'''
Pull the data from webanno and prep it for modelling.
1. Convert the json into labels
2. Tokenize the source documents
3.  Create a data frame:
    1.  token
    2.  token lenght, start, and end
    3.  columns for each of the tags.  they'll span [-1,0, 1], and we can convert them to categorical later

'''

import pandas as pd
from _99_project_module import read_txt, read_json, process_webanno_output
import os
import re
import spacy

pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000


# load the spacy stuff
nlp = spacy.load("en", disable=['parser', 'tagger', 'ner'])

# stepping through the files, use the spacy nlp function to build a data frame of tokens and their spans
tags = ['Functionalimpairment', "Msk_prob", "Nutrition", "Resp_imp", 'Fall_risk']
mapping_dict = dict(frailty_nos="Functionalimpairment",
                    msk_prob_tags="Msk_prob",
                    nutrition="Nutrition",
                    resp_imp_tags="Resp_imp",
                    fall_risk_tags="Fall_risk")


def tokenize_and_label(anno_dir, webanno_output, annotator_of_record):
    webanno_unzipped_dir = re.sub("\.zip", "", webanno_output)
    if webanno_unzipped_dir not in os.listdir(anno_dir):
        process_webanno_output(anno_dir, webanno_output)
        os.system(f"mkdir {anno_dir + webanno_unzipped_dir}/labels/")

    for stub in os.listdir(anno_dir + webanno_unzipped_dir + "/annotation"):
        print(stub)
        # get the original note and tokenize it
        note = read_txt(f"{anno_dir + webanno_unzipped_dir}/source/{stub}")
        res = nlp(note)
        span_df = pd.DataFrame([{"token": i.text, 'length': len(i.text_with_ws)} for i in res])
        span_df['end'] = span_df.length.cumsum().astype(int)
        span_df['start'] = span_df.end - span_df.length
        assert int(span_df.end[-1:]) == len(note)
        # merge them on
        for i in tags:
            span_df[i] = 0

        # get the annotation file and process it into something that I can work with
        anno = read_json(f"{anno_dir + webanno_unzipped_dir}/annotation/{stub}/{annotator_of_record}.json")
        l = []
        for i in tags:
            try:
                l.append(pd.DataFrame(anno["_views"]["_InitialView"][i]))
            except Exception:
                pass
        if len(l) > 0:  # only do this if there are any actual tags
            tag_df = pd.concat(l, sort=True)
            tag_df.drop(columns='sofa', inplace=True)
            tag_df = pd.melt(tag_df, id_vars=['begin', 'end'])
            tag_df = tag_df[~tag_df.value.isnull()]
            tag_df.value = [1 if 'yes' in i else -1 for i in tag_df.value]
            for i in range(tag_df.shape[0]):
                var = mapping_dict[tag_df.variable.iloc[i]]
                span_df.loc[(span_df.end > tag_df.begin.iloc[i]) &
                            (span_df.start < tag_df.end.iloc[i]), var] = tag_df.value.iloc[i]
        # save the file
        outfn = re.sub('\.txt', '_labels.pkl', stub)
        span_df.to_pickle(f"{anno_dir + webanno_unzipped_dir}/labels/{outfn}")


tokenize_and_label(f'{os.getcwd()}/annotation/', "frailty_phenotype_batch_1_2020-02-05_1016.zip", 'jake')

