from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Folosim panda pentru citirea csv-ului si pentru a crea dataframe-uls

# Citirea datelor din fisierul csv si sapmpling-ul acestora
file_read = pd.read_csv("nitro-language-processing-2/train_data.csv")
# Luam 100% din datele din csv, adica frac = 1
dataframe = file_read.sample(frac = 1)
# print(dataframe.to_string())

# Preprocesare: eliminarea semnelor de punctuatie, spatii, transformarea in litere mici
dataframe['Text'] = dataframe['Text'].str.replace('[^\w\s]', '', regex = True)
dataframe['Text'] = dataframe['Text'].str.lower()

text = dataframe['Text']
label = dataframe['Final Labels']

print(label.value_counts())

# dataset_proportion = 0.25;
#
# text_train, text_test, label_train, label_test = train_test_split(text, label, test_size = dataset_proportion,
#                                                                   shuffle = True, stratify = label)
#
# pipe = Pipeline(steps = [('vectorize', CountVectorizer(ngram_range = (1, 1), token_pattern = r'\b\w+\b')),
#                          ('classifier', MultinomialNB())])
# pipe.fit(text_train, label_train)
#
# label_predict = pipe.predict(text_test)
#
# print(f"Acuratete: ", accuracy_score(label_test, label_predict))