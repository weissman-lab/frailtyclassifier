'''
Picking up from ACD pulling files from webanno & converting JSON file into labels
This script will tokenize the source documents & create a word embedding matrix
'''

import pandas as pd
from gensim.models import Word2Vec
import os

pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000

anno_dir = f'{os.getcwd()}/annotation/'
os.listdir(anno_dir)


# # unzip the output file
webanno_output = 'ACD_sandbox_2020-01-27_1007.zip'
annotator_of_record = 'andrew'
webanno_unzipped_dir = re.sub("\.zip", "", webanno_output)
if webanno_unzipped_dir not in os.listdir(anno_dir):
    process_webanno_output(anno_dir, webanno_output)
    os.system(f"mkdir {anno_dir+webanno_unzipped_dir}/labels/")


# load the spacy stuff
nlp = spacy.load("en", disable=['parser', 'tagger', 'ner'])


# stepping through the files, use the spacy nlp function to build a data frame of tokens and their spans
tags = ['Functionalimpairment', "Msk_prob", "Nutrition", "Resp_imp", 'Fall_risk']
mapping_dict = dict(frailty_nos = "Functionalimpairment",
                    msk_prob_tags = "Msk_prob",
                    nutrition = "Nutrition",
                    resp_imp_tags = "Resp_imp",
                    fall_risk_tags = "Fall_risk")
for stub in os.listdir(anno_dir+webanno_unzipped_dir+"/annotation"):
    # get the annotation file and process it into something that I can work with
    anno = read_json(f"{anno_dir+webanno_unzipped_dir}/annotation/{stub}/{annotator_of_record}.json")
    l = []
    for i in tags:
        try:
            l.append(pd.DataFrame(anno["_views"]["_InitialView"][i]))
        except Exception:
            pass
    tag_df = pd.concat(l, sort = True)
    tag_df.drop(columns='sofa', inplace=True)
    tag_df = pd.melt(tag_df, id_vars=['begin', 'end'])
    tag_df = tag_df[~tag_df.value.isnull()]
    tag_df.value = [1 if 'yes' in i else -1 for i in tag_df.value]
    # get the original note and tokenize it