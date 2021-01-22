'''
Modifies the following two functions from _05_ingest_annotations.py in order to
label sentences using scispaCy before the text is processed (e.g. lowercase,
remove newlines, etc.):
tokenize_and_label -> tokenize_and_label_sent
featurize -> featurize_sent
'''

import multiprocessing as mp
import os
import re
import warnings

import numpy as np
import pandas as pd
import scispacy
import spacy
from configargparse import ArgParser
from gensim.models import KeyedVectors

# import annotation ingestion scripts from the prior pipeline that did not
# include sentences
# from _05_ingest_annotations import process_webanno_output, embeddings_catcher, \
#     remove_headers
from _99_project_module import read_txt, read_json, ncol

pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000



def process_webanno_output(output_file_path):
    output_dir = re.sub('\.zip', '', output_file_path)
    os.system(f"mkdir {output_dir}")
    os.system(f'unzip -o {output_file_path} -d {output_dir}')
    # unzip all of the annotation files in the overall output file
    for i in os.listdir(output_dir + "/annotation"):  # note that the dirs are named after the text files
        cmd = f"unzip -n {output_dir}/annotation/{i}/\*.zip -d {output_dir}/annotation/{i}/"
        os.system(cmd)
    # same with curation
    for i in os.listdir(output_dir + "/curation"):  # note that the dirs are named after the text files
        cmd = f"unzip -n {output_dir}/curation/{i}/\*.zip -d {output_dir}/curation/{i}/"
        os.system(cmd)

# removed this from global env (in case it interferes with scispacy library in _05_ingest_to_sentence.py
# nlp = spacy.load("en", disable=['parser', 'tagger', 'ner'])
#
# # stepping through the files, use the spacy nlp function to build a data frame of tokens and their spans
# tags = ['Frailty_nos', "Msk_prob", "Nutrition", "Resp_imp", 'Fall_risk']
# mapping_dict = dict(frailty_nos_tags="Frailty_nos",
#                     msk_prob_tags="Msk_prob",
#                     nutrition="Nutrition",
#                     resp_imp_tags="Resp_imp",
#                     fall_risk_tags="Fall_risk")

def remove_headers(fi):
    '''
    This removes metadata tags that are bounded by long sequences of dashes
    It'll loop through the tokens and make note of places with long such strings.
    Then it'll check if there is an even number of them.
    If there is an even number, it will define spans.
    Inside each span it'll assert that there is something that looks like a timestamp
    If all that is true, those obs will be removed
    '''
    # make sure that all of the separators are the correct number of dashes
    dash_token_indices = [fi.index[i] for i in fi.index if fi.token[i] == '--------------------------------------------------------------']
    # make sure that there is an even number of them
    assert len(dash_token_indices) %2 == 0
    # make sure that they all have the same distance apart
    spanlengths = [dash_token_indices[i] - dash_token_indices[i-1] for i in range(1, len(dash_token_indices),2)]
    if len(list(set(spanlengths))) != 1:
        breakpoint()
    assert(len(list(set(spanlengths)))) == 1
    # make sure there is at least one year inside these spans
    for i in range(1, len(dash_token_indices),2):
        jstring = "".join(fi.token[(dash_token_indices[i-1]):(dash_token_indices[i]+1)])
        assert re.search("(19[789]\d|20[012]\d)", jstring)
    # if these pass, lose the spans
    for i in range(1, len(dash_token_indices),2):
        fi = fi.drop(index=list(range((dash_token_indices[i-1]-1),(dash_token_indices[i]+2))))
    # now drop the newlines -- they're not in the dictionary
    fi = fi[fi.token != "\n"]
    fi = fi[fi.token != "\n "]
    return fi


def embeddings_catcher(tok, embeddings):
    '''
    This function is a workaround for the fact that some OOV words throw errors in the OA corpus, because the
    dictionary wasn't identical to the corpus.
    Tok is a string and embeddings is a gensim object
    '''
    try:
        return embeddings[tok]
    except Exception:
        return np.zeros(embeddings.vector_size)

def set_custom_boundaries(doc):
    # carriage return is a sentence boundary
    newline = re.compile("\n")
    # a period inside a multi-word expressions is a sentence boundary
    mwe_period_space = re.compile(r"\._")
    for token in doc[:-1]:
        if newline.search(token.text) is not None:
            doc[token.i + 1].is_sent_start = True
        if mwe_period_space.search(token.text) is not None:
            doc[token.i + 1].is_sent_start = True
    return doc

#modified 'tokenize_and_label' and 'featurize' to output sentences
def tokenize_and_label_sent(output_file_path,
                            annotator_of_record="CURATION_USER"):
    webanno_unzipped_dir = re.sub("\.zip", "", output_file_path)
    os.system(f"mkdir {webanno_unzipped_dir}/labels/")
    # load scispacy model
    sci_nlp = spacy.load("en_core_sci_md", disable=['tagger', 'ner'])
    # add custom boundary to nlp pipeline
    sci_nlp.add_pipe(set_custom_boundaries, before="parser",
                     name='set_custom_boundaries')

    # stepping through the files, use the spacy nlp function to build a data frame of tokens and their spans
    tags = ['Frailty_nos', "Msk_prob", "Nutrition", "Resp_imp", 'Fall_risk']
    mapping_dict = dict(frailty_nos_tags="Frailty_nos",
                        msk_prob_tags="Msk_prob",
                        nutrition="Nutrition",
                        resp_imp_tags="Resp_imp",
                        fall_risk_tags="Fall_risk")
    outlist = []
    stubs = [i for i in os.listdir(webanno_unzipped_dir + "/curation") if
             '.txt' in i]  # do this to avoid crufty DS_store files getting in there
    for stub in stubs:
        print(stub)
        # get the original note and tokenize it
        note = read_txt(f"{webanno_unzipped_dir}/source/{stub}")
        res = sci_nlp(note)
        span_df = pd.DataFrame([{"token": i.text,
                                 'length': len(i.text_with_ws),
                                 'sent_start': i.is_sent_start} for i in res])
        # number each sentence (to maintain sentence boundaries after cleaning up tokens)
        sentence = []
        sent = -1
        for s in span_df['sent_start']:
            if s is True:
                sent += 1
            sentence.append(sent)
        span_df['sentence'] = sentence
        span_df['end'] = span_df.length.cumsum().astype(int)
        span_df['start'] = span_df.end - span_df.length
        assert int(span_df.end[-1:]) == len(note)
        # merge them on
        for i in tags:
            span_df[i] = 0

        # get the annotation file and process it into something that I can work with
        anno = read_json(
            f"{webanno_unzipped_dir}/curation/{stub}/{annotator_of_record}.json")
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
                            (span_df.start < tag_df.end.iloc[i]), var] = \
                    tag_df.value.iloc[i]
            # add important variables
            span_df['note'] = re.sub(".txt", "", stub)
            # span_df['month'] = span_df.note.apply(lambda x: int(x.split("_")[2][1:]))
            span_df['month'] = span_df.note.apply(
                lambda x: int(re.sub("m", "", x.split("_")[-2])))
            span_df['PAT_ID'] = span_df.note.apply(lambda x: x.split("_")[-1])
            outlist.append(span_df)
    return outlist


def featurize_sent(file,
                   # the data frame -- a product of the tokenize_and_label function
                   embeddings,
                   # this is either the path to the embeddings (in which case they will be loaded) or the object name of the loaded embeddings
                   ):  # if idx is not None, this is how many random indices to pull
    # First check and see whether the embeddings object is loaded
    if isinstance(embeddings, str):  # load it if it's not
        embeddings = KeyedVectors.load(embeddings, mmap='r')
    assert ("Word2Vec" in type(embeddings).__name__) or (
            "FastText" in type(embeddings).__name__)
    # now load the file and remove its headers (note concatenation indicators)
    # fi = pd.read_pickle(f"{anno_dir + webanno_output}/labels/{file}")
    print("a")
    fi = remove_headers(file)
    # fix sentence numbering now that we have cleaned up the tokens
    sentence = []
    sent = -1
    for s in range(fi.shape[0]):
        if fi.iloc[s]['sentence'] != fi.iloc[s - 1]['sentence']:
            sent += 1
        sentence.append(sent)
    fi['sentence'] = sentence
    # add metadata
    fi['index'] = fi.index
    # lowercase the words
    fi.token = fi.token.str.lower()
    # loop through the words and make an embeddings matrix
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Elist = [embeddings_catcher(i, embeddings) for i in fi.token]
    print("b")
    # make a variable that indicates whether the word was found in the vocab
    fi['invocab'] = [1 if len(set(Elist[i])) > 1 else 0 for i in
                     range(len(Elist))]
    # loop through the functions and apply them to the list of embeddings
    outlist = []
    # identity
    Emat = np.vstack(Elist)
    outlist.append(pd.DataFrame(data=Emat,
                                index=fi.index.tolist(),
                                columns=["identity_" + str(i) for i in
                                         range(ncol(Emat))]))

    # construct output
    output = pd.concat([fi] + outlist, axis=1)
    return output.reset_index(drop=True)


def main():
    p = ArgParser()
    p.add("-z", "--zipfile", help="zip file to ingest")
    p.add("-e", "--embeddings", help="path to the embeddings file")
    p.add("-o", "--outdir", help="path to save embedded notes")
    p.add("-s", "--structured_data_path", help="path to structured data")
    options = p.parse_args()
    zipfile = options.zipfile
    embeddings = options.embeddings
    outdir = options.outdir
    structured_data_path = options.structured_data_path
    # structured data before imputation: /output/structured_data_merged_cleaned.csv
    strdat = pd.read_csv(structured_data_path,
                          dtype={'SEX': 'string',
                                 'MARITAL_STATUS': 'string',
                                 'RELIGION': 'string',
                                 'EMPY_STAT': 'string',
                                 'RACE': 'string',
                                 'LANGUAGE': 'string'})
    strdat.drop(columns="Unnamed: 0", inplace=True)
    # unzip the raw output
    process_webanno_output(zipfile)
    # tokenize
    dflist = tokenize_and_label_sent(zipfile)
    # merge on the structured data
    dflist = [i.merge(strdat, how="left") for i in dflist]
    # embed
    pool = mp.Pool(
        8)  # hard-coding 8 because the embeddings takes a ton of memory
    embedded_notes = pool.starmap(featurize_sent, [(i, embeddings) for i in dflist])
    pool.close()
    # save them all
    for i in embedded_notes:
        i.to_csv(f"{outdir}enote_{str(i.note.iloc[0])}.csv")


if __name__ == "__main__":
    main()

# azure
# python _05_ingest_to_sentence.py -z /share/gwlab/frailty/annotation/frailty_phenotype_AL_00_2020-06-29_0939.zip -e /share/acd-azure/pwe/output/built_models/OA_ALL/W2V_300/w2v_oa_all_300d.bin -o /share/gwlab/frailty/output/notes_labeled_embedded_SENTENCES/ -s /share/gwlab/frailty/output/impdat_dums.csv && python _05_ingest_to_sentence.py -z /share/gwlab/frailty/annotation/frailty_phenotype_AL_01_2020-08-13_1218.zip -e /share/acd-azure/pwe/output/built_models/OA_ALL/W2V_300/w2v_oa_all_300d.bin -o /share/gwlab/frailty/output/notes_labeled_embedded_SENTENCES/ -s /share/gwlab/frailty/output/impdat_dums.csv && python _05_ingest_to_sentence.py -z /share/gwlab/frailty/annotation/frailty_phenotype_AL_01_ADDENDUM_2020-08-13_1218.zip -e /share/acd-azure/pwe/output/built_models/OA_ALL/W2V_300/w2v_oa_all_300d.bin -o /share/gwlab/frailty/output/notes_labeled_embedded_SENTENCES/ -s /share/gwlab/frailty/output/impdat_dums.csv && python _05_ingest_to_sentence.py -z /share/gwlab/frailty/annotation/frailty_phenotype_batch_1_2020-03-02_1328.zip -e /share/acd-azure/pwe/output/built_models/OA_ALL/W2V_300/w2v_oa_all_300d.bin -o /share/gwlab/frailty/output/notes_labeled_embedded_SENTENCES/ -s /share/gwlab/frailty/output/impdat_dums.csv && python _05_ingest_to_sentence.py -z /share/gwlab/frailty/annotation/frailty_phenotype_batch_2_2020-03-02_1325.zip -e /share/acd-azure/pwe/output/built_models/OA_ALL/W2V_300/w2v_oa_all_300d.bin -o /share/gwlab/frailty/output/notes_labeled_embedded_SENTENCES/ -s /share/gwlab/frailty/output/impdat_dums.csv && python _05_ingest_to_sentence.py -z /share/gwlab/frailty/annotation/frailty_phenotype_batch_3_2020-04-18_1444.zip -e /share/acd-azure/pwe/output/built_models/OA_ALL/W2V_300/w2v_oa_all_300d.bin -o /share/gwlab/frailty/output/notes_labeled_embedded_SENTENCES/ -s /share/gwlab/frailty/output/impdat_dums.csv

# mac
# python _05_ingest_to_sentence.py -z /Users/martijac/Documents/Frailty/frailty_classifier/annotation/frailty_phenotype_AL_00_2020-06-29_0939.zip -e /Users/martijac/Documents/Frailty/frailty_classifier/embeddings/W2V_300_all/w2v_oa_all_300d.bin -o /Users/martijac/Documents/Frailty/frailty_classifier/output/notes_labeled_embedded_SENTENCES/ -s /Users/martijac/Documents/Frailty/frailty_classifier/output/structured_data_merged_cleaned.csv && python _05_ingest_to_sentence.py -z /Users/martijac/Documents/Frailty/frailty_classifier/annotation/frailty_phenotype_AL_01_2020-08-13_1218.zip -e /Users/martijac/Documents/Frailty/frailty_classifier/embeddings/W2V_300_all/w2v_oa_all_300d.bin -o /Users/martijac/Documents/Frailty/frailty_classifier/output/notes_labeled_embedded_SENTENCES/ -s /Users/martijac/Documents/Frailty/frailty_classifier/output/structured_data_merged_cleaned.csv && python _05_ingest_to_sentence.py -z /Users/martijac/Documents/Frailty/frailty_classifier/annotation/frailty_phenotype_AL_01_ADDENDUM_2020-08-13_1218.zip -e /Users/martijac/Documents/Frailty/frailty_classifier/embeddings/W2V_300_all/w2v_oa_all_300d.bin -o /Users/martijac/Documents/Frailty/frailty_classifier/output/notes_labeled_embedded_SENTENCES/ -s /Users/martijac/Documents/Frailty/frailty_classifier/output/structured_data_merged_cleaned.csv && python _05_ingest_to_sentence.py -z /Users/martijac/Documents/Frailty/frailty_classifier/annotation/frailty_phenotype_batch_1_2020-03-02_1328.zip -e /Users/martijac/Documents/Frailty/frailty_classifier/embeddings/W2V_300_all/w2v_oa_all_300d.bin -o /Users/martijac/Documents/Frailty/frailty_classifier/output/notes_labeled_embedded_SENTENCES/ -s /Users/martijac/Documents/Frailty/frailty_classifier/output/structured_data_merged_cleaned.csv && python _05_ingest_to_sentence.py -z /Users/martijac/Documents/Frailty/frailty_classifier/annotation/frailty_phenotype_batch_2_2020-03-02_1325.zip -e /Users/martijac/Documents/Frailty/frailty_classifier/embeddings/W2V_300_all/w2v_oa_all_300d.bin -o /Users/martijac/Documents/Frailty/frailty_classifier/output/notes_labeled_embedded_SENTENCES/ -s /Users/martijac/Documents/Frailty/frailty_classifier/output/structured_data_merged_cleaned.csv && python _05_ingest_to_sentence.py -z /Users/martijac/Documents/Frailty/frailty_classifier/annotation/frailty_phenotype_batch_3_2020-04-18_1444.zip -e /Users/martijac/Documents/Frailty/frailty_classifier/embeddings/W2V_300_all/w2v_oa_all_300d.bin -o /Users/martijac/Documents/Frailty/frailty_classifier/output/notes_labeled_embedded_SENTENCES/ -s /Users/martijac/Documents/Frailty/frailty_classifier/output/structured_data_merged_cleaned.csv

# grace
# python _05_ingest_to_sentence.py -z /home/jakem/frailty_classifier/annotation/frailty_phenotype_AL_01_ADDENDUM_2020-08-13_1218.zip -e /home/jakem/frailty_classifier/embeddings/W2V_300_all/w2v_oa_all_300d.bin -o /home/jakem/frailty_classifier/output/notes_labeled_embedded_SENTENCES/ -s /home/jakem/frailty_classifier/output/impdat_dums.csv
