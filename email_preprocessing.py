import keras
import os,datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import nltk
import matplotlib.pyplot as plt


df = pd.read_csv("email.csv")
df['date']=pd.to_datetime(df['date'])

# Define the function to categorize time frames
def categorize_time_frame(hour):
    if 0 <= hour < 6:
        return 0
    elif 6 <= hour < 12:
        return 1
    elif 12 <= hour < 18:
        return 2
    else:
        return 3

# Apply the function to create the 'time_frame' column
new_df = df.copy()
new_df['time_frame'] = df['date'].dt.hour.apply(categorize_time_frame)
new_df['day'] = df['date'].dt.day
new_df['month'] = df['date'].dt.month
new_df['year'] = df['date'].dt.year

new_df=new_df.drop(columns="date")


le_user = LabelEncoder()
le_user.fit(new_df['user'])
new_df['user'] = le_user.transform(new_df['user'])


#Save the user label encoder
pkl_user_output = open("user_encoder.pkl",'wb')
pickle.dump(le_user, pkl_user_output)


le_pc = LabelEncoder()
le_pc.fit(new_df['pc'])
new_df['pc'] = le_pc.transform(new_df['pc'])

#Save the PC label encoder
pkl_PC_output = open("pc_encoder.pkl",'wb')
pickle.dump(le_pc, pkl_PC_output)

new_df.to_csv("processed_email.csv", index=False)
