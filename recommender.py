#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 13:20:37 2020

@author: shivam
Data Set https://grouplens.org/datasets/movielens/
"""

import pandas as pd

r_cols = ['userId', 'movieId', 'rating']
ratings = pd.read_csv('/home/shivam/Desktop/mtech 4/ml-latest-small/ratings.csv', sep=',', names=r_cols, usecols=range(3), skiprows = [0])
ratings.head()

m_cols = ['movieId', 'title']
movies = pd.read_csv('/home/shivam/Desktop/mtech 4/ml-latest-small/movies.csv',sep=',',names=m_cols, usecols=range(2), skiprows = [0])
movies.head()

ratings = pd.merge(movies, ratings)
ratings.head()

userRatings = ratings.pivot_table(index=['userId'], columns=['title'], values='rating')
userRatings.head()

corrMatrix = userRatings.corr(method='pearson', min_periods=100)
corrMatrix.head()

myRatings = userRatings.loc[1].dropna()
myRatings.head()

simCandidates = pd.Series()
for i in range(0, len(myRatings.index)):
    print("Adding sims for " + myRatings.index[i] + "...")
    sims = corrMatrix[myRatings.index[i]].dropna()
    sims = sims.map(lambda x: x * myRatings[i])
    simCandidates = simCandidates.append(sims)
    
print("Sorting...")
simCandidates.sort_values(inplace = True, ascending = False)
print(simCandidates.head(10))


simCandidates = simCandidates.groupby(simCandidates.index).sum()

simCandidates.sort_values(inplace = True, ascending = False)
simCandidates.head(10)

filteredSims = simCandidates.drop(myRatings.index, errors='ignore')
filteredSims.head(10)


