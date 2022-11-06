# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 12:03:00 2022

@author: Gopinaath
"""
# importing the pandas library
import pandas as pd

# import Dataset 
df = pd.read_csv("C:\\Users\\Gopinaath\\OneDrive\\Documents\\Deployment files 5\\Recommendedwithoutresult.csv")

# Getting the file information
df.info()

# Finding the null value in the column
df.isna().sum()

#Finding the shape of dataset
df.shape

#finding the column names
df.columns

#Finding the items in main columns
df.MainIngredient

df.HighlypreferredIngredient

df.LowpreferredIngredient

df.PreferredMealType

df.PreferredFoodType

#Checking the datatypes of the column value
df.dtypes

#term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus
from sklearn.feature_extraction.text import TfidfVectorizer

# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words = "english")

# Preparing the Tfidf matrix by fitting and transforming
tfidf_matrix = tfidf.fit_transform(df['MainIngredient']+df['PreferredMealType']+df['PreferredFoodType']+df['HighlypreferredIngredient']+df['LowpreferredIngredient'])   #Transform a count matrix to a normalized tf or tf-idf representation


tfidf_matrix.shape

#importing the linear kernel library
from sklearn.metrics.pairwise import linear_kernel

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

# creating a mapping of df name to index number
df_index = pd.Series(df.index, index = df['Dish']).drop_duplicates()

#Checking whether the below code shows the column index no
df_id = df_index["Apple Rabdi"]

df_id

def get_recommendations(Dish, topN):    
    # topN = 10
    # Getting the movie index using its title 
    df_id = df_index[Dish]
    
    # Getting the pair wise similarity score for all the anime's with that 
    # anime
    cosine_scores = list(enumerate(cosine_sim_matrix[df_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar movies 
    cosine_scores_N = cosine_scores[0: topN+1]
    
    # Getting the movie index 
    df_idx  =  [i[0] for i in cosine_scores_N]
    df_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar movies and scores
    df_similar_show = pd.DataFrame(columns=["Dish", "Score"])
    df_similar_show["Dish"] = df.loc[df_idx, "Dish"]
    df_similar_show["Score"] = df_scores
    df_similar_show.reset_index(inplace = True)  
    print (df_similar_show)
    # The End


df_index["Aata Cake"]
get_recommendations("Chicken Malai Seekh Kebab", topN = 10)


#import pickle

# Saving model to disk
#pickle.dump(df, open('recommendation.pkl','wb'))

#df["Dish"].values

#df.to_dict()

#pickle.dump(df.to_dict(), open('recommendation.pkl','wb'))

#pickle.dump(cosine_sim_matrix, open('similarity.pkl','wb'))

#df.iloc[18].Dish
