import pandas as pd
import nltk
from nltk.tokenize import TweetTokenizer
import re
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy import sparse


# tmp = [subdir for _,subdir,_ in os.walk('./data')]
# selected_problem = [int(p) for l in tmp for p in l]

# Token
tknzr = TweetTokenizer()
# Stop Words
stopwords = nltk.corpus.stopwords.words("english")
stopwords = set(stopwords)
# Stemming
english_stemmer = nltk.stem.PorterStemmer()
#Tf-idf Vectorize
vectorizer = TfidfVectorizer()
problem = pd.read_csv('./data/selected_problem_30')
problem.fillna(' ',inplace=True)
#Filter Problem
# mask = problem['id'].apply(lambda x:x in selected_problem)
# problem = problem[mask]
problem['description'] = (problem['description'] +' '+ problem['input'] +' '+ problem['output']).apply(
    lambda s:re.sub(r'[^a-zA-Z0-9<=>\+\-/\s]',repl=' ',string=s))
problem['description'] = problem['description'].apply(tknzr.tokenize)
problem['description'] = problem['description'].apply(lambda x:' '.join([english_stemmer.stem(s.lower()) for s in x]))
vector = vectorizer.fit_transform(problem['description'].values)
X = vector.toarray()
pca = PCA(n_components=29)
newX = pca.fit_transform(X)
to_save = dict()
for i in range(problem.shape[0]):
    to_save[str(problem['id'][i])] = newX[i].reshape(1,-1).tolist()
js = json.dumps(to_save)
with open('./data/tf_idf_vector.txt','w') as f:
    f.write(js)
print(pca.explained_variance_ratio_)
print(cosine_similarity(to_save['1000'],to_save['1001']))
print(cosine_similarity(to_save['1000'],to_save['1002']))
print(cosine_similarity(to_save['1000'],to_save['1003']))
