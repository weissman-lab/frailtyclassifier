'''
Picking up from ACD pulling files from webanno & converting JSON file into labels
This script will tokenize the source documents & create a word embedding matrix
'''

import pandas as pd
from _99_project_module import read_txt, read_json, process_webanno_output
import os
import re
import spacy
import gensim
from gensim.models import Word2Vec

pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000

anno_dir = f'{os.getcwd()}/annotation/'
os.listdir(anno_dir)


# unzip the output file
webanno_output = 'ACD_sandbox_2020-01-29_0847.zip'
annotator_of_record = 'andrew'
webanno_unzipped_dir = re.sub('\.zip', "", webanno_output)
test_dir = re.sub('/labels', "", anno_dir+webanno_unzipped_dir)
if webanno_unzipped_dir not in os.listdir(anno_dir):
    process_webanno_output(anno_dir, webanno_output)
    os.system(f"mkdir {anno_dir+webanno_unzipped_dir}/labels/")


# load the spacy stuff
# note: may need to load spacy model en with this script in terminal: [full path to python interpreter] -m spacy download en
nlp = spacy.load("en", disable=['parser', 'tagger', 'ner'])


# stepping through the files, use the spacy nlp function to build a data frame of tokens and their spans
tags = ['Functionalimpairment', "Msk_prob", "Nutrition", "Resp_imp", 'Fall_risk']
mapping_dict = dict(frailty_nos = "Functionalimpairment",
                    msk_prob_tags = "Msk_prob",
                    nutrition = "Nutrition",
                    resp_imp_tags = "Resp_imp",
                    fall_risk_tags = "Fall_risk")

print(anno_dir) #/Users/jmartin89/Documents/Frailty/frailty_classifier/annotation/
print(webanno_unzipped_dir) #ACD_sandbox_2020-01-27_1007
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
    note = read_txt(f"{anno_dir + webanno_unzipped_dir}/source/{stub}")
    res = nlp(note)
    span_df = pd.DataFrame([{"token": i.text, 'length': len(i.text_with_ws)} for i in res])
    span_df['end'] = span_df.length.cumsum().astype(int)
    span_df['start'] = span_df.end - span_df.length
    assert int(span_df.end[-1:]) == len(note)
    # merge them on
    for i in tags:
        span_df[i] = 0
    for i in range(tag_df.shape[0]):
        var = mapping_dict[tag_df.variable.iloc[i]]
        span_df.loc[(span_df.end > tag_df.begin.iloc[i]) &
                    (span_df.start < tag_df.end.iloc[i]), var] = tag_df.value.iloc[i]
    outfn = re.sub('\.txt', '_labels.pkl', stub)
    span_df.to_pickle(f"{anno_dir + webanno_unzipped_dir}/labels/{outfn}")

#convert gensim word2vec into format usable by spacy
from gensim.models import Word2Vec, keyedvectors
word2vec = Word2Vec.load('/Users/jmartin89/Documents/Frailty/JMworkingproj/W2V_300/w2v_OA_CR_300d.bin')
word2vec.wv.save_word2vec_format('w2v_test.txt')
# code for terminal: python -m spacy init-model en /Users/jmartin89/Documents/Frailty/frailty_classifier/spacy/w2v_300 --vectors-loc /Users/jmartin89/Documents/Frailty/frailty_classifier/w2v_test.txt
word2vec_load = spacy.load('/Users/jmartin89/Documents/Frailty/frailty_classifier/spacy/w2v_300')
#load the note from above with word2vec as language
w2v_note = word2vec_load(note)
#test print the tokens
for token in w2v_note:
    print(token)
#test print the token vectors
for token in w2v_note:
    print('Vector for %s:' % token, token.vector)