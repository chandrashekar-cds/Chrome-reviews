# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 10:54:51 2022

@author: Chandrashekar
"""

# to deploy the chrome review data output on streamlit

# import module
import streamlit as st
# Title for streamlit
st.title("Chrome Reviews - positive reviews with low ratings")


import pandas as pd


df = pd.read_csv('chrome_reviews.csv')
#df.head()



import numpy as np
df_NA = df.dropna(how = 'all')
#df[df['Star'] != 3]
#df_NA.keys()
df_NA['Positivity'] = np.where(df_NA['Star'] > 3, 1, 0)
cols = ['ID', 'Star', 'Review URL', 'Thumbs Up', 'User Name', 'Developer Reply', 'Version','Review Date', 'App ID']
df_NA.drop(cols, axis=1, inplace=True)

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# Initialize empty array
# to append clean text
corpus = []
num = len(df_NA)

df_NA['Text'] = df_NA['Text'].astype(str)
# (reviews) rows to clean
for i in range(0, num):
	
	# column : "Text", row ith
	review = re.sub('[^a-zA-Z]', ' ', df_NA['Text'][i])
	
	# convert all cases to lower cases
	review = review.lower()
	
	# split to array(default delimiter is " ")
	review = review.split()
	
	# creating PorterStemmer object to
	# take main stem of each word
	ps = PorterStemmer()
	
	# loop for stemming each word
	# in string array at ith row
	review = [ps.stem(word) for word in review
				if not word in set(stopwords.words('english'))]
				
	# rejoin all string array elements
	# to create back into a string
	review = ' '.join(review)
	
	# append each string to create
	# array of clean text
	corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1000)
X = cv.fit_transform(corpus).toarray()
y = df_NA.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 350,criterion = 'entropy')
                             
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


from sklearn.metrics import confusion_matrix
 
cm = confusion_matrix(y_test, y_pred)



uploaded_file = st.file_uploader("Choose a file for checking review/rating discrepancy")
st.write("Waiting for input")
if uploaded_file is not None:
    
     # Can be used wherever a "file-like" object is accepted:
     df_test = pd.read_csv(uploaded_file)
     #df_iloc = dataframe.iloc[3] 
     
     #st.write(dataframe['Text'][3],dataframe['Star'][3])
#i = input("Press Enter to continue: ") 

df_tNA = df_test.dropna(how = 'all')
df_tNA['Positivity'] = np.where(df_tNA['Star'] > 3, 1, 0)
cols = ['ID', 'Review URL', 'Thumbs Up', 'User Name', 'Developer Reply', 'Version','Review Date', 'App ID']
df_tNA.drop(cols, axis=1, inplace=True)



# NLP stuff on test file
corpus_test = []
num = len(df_tNA)

df_tNA['Text'] = df_tNA['Text'].astype(str)
# (reviews) rows to clean
for i in range(0, num):
	
	# column : "Text", row ith
	review = re.sub('[^a-zA-Z]', ' ', df_tNA['Text'][i])
	
	# convert all cases to lower cases
	review = review.lower()
	
	# split to array(default delimiter is " ")
	review = review.split()
	
	# creating PorterStemmer object to
	# take main stem of each word
	ps = PorterStemmer()
	
	# loop for stemming each word
	# in string array at ith row
	review = [ps.stem(word) for word in review
				if not word in set(stopwords.words('english'))]
				
	# rejoin all string array elements
	# to create back into a string
	review = ' '.join(review)
	
	# append each string to create
	# array of clean text
	corpus_test.append(review)
#print(corpus_test)

cv_test = CountVectorizer(max_features = 1000)
X_testr = cv_test.fit_transform(corpus_test).toarray()
y_testr = df_tNA.iloc[:, 2].values

y_predr = model.predict(X_testr)
cm = confusion_matrix(y_testr, y_predr)

st.write("The list of reviews where the reviews and ratings probably don't match are as below")
for i in range(0, len(df_tNA)):
    if(y_predr[i] == 1 and y_testr[i] == 0):
            st.write(df_tNA['Text'][i],df_tNA['Star'][i])
            #count = count + 1
#st.write("Hi")
