import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

df_train = pd.read_csv('labeledTrainData.tsv', header=0, sep='\t')
print(df_train.head())

example1 = BeautifulSoup(df_train['review'][0])
# print(example1)
letters = re.sub("[^A-Za-z]", " ", example1.get_text())
# print(letters)
lower_case = letters.lower()
words = lower_case.split()
# print(words)
words = [w for w in words if w not in stopwords.words('english')]
# print(words)
def rev2words(raw_review):
    example1 = BeautifulSoup(raw_review)
    letters = re.sub("[^A-Za-z]", " ", example1.get_text())
    lower_case = letters.lower()
    words = lower_case.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return(" ".join(words))