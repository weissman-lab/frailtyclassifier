'''
File purpose...
'''

import pandas as pd
from gensim.models import Word2Vec
import os

# import
datadir = '/Users/jmartin89/Documents/Frailty/JMworkingproj/data/'
outdir = '/Users/jmartin89/Documents/Frailty/JMworkingproj/output/'
figdir = '/Users/jmartin89/Documents/Frailty/JMworkingproj/figures/'

# preferences
pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000

# load the raw notes
os.listdir('/Users/jmartin89/Documents/Frailty/JMworkingproj/output/random_pull')


df = pd.read_pickle(f"{outdir}raw_notes_df.pkl")

#load embeddings

model = Word2Vec.load('W2V_300/w2v_OA_CR_300d.bin')