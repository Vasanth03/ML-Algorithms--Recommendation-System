# import os
import pandas as pd

# import Dataset 
game = pd.read_csv("copy path", encoding = 'utf8')
game.shape # shape
game.columns
game.describe()
game.info()


# EDA
no_ratings = len(game) #Total number of ratings in the gn dataset
no_games   = len(game['game'].unique())
no_users   = len(game['userId'].unique())


print(f"Number of ratings: {no_ratings}")
print(f"Number of unique games: {no_games}")
print(f"Number of unique users: {no_users}")
print(f"Average ratings per user: {round(no_ratings/no_users, 2)}")
print(f"Average ratings per movie: {round(no_ratings/no_games, 2)}")
  
user_freq = game[['userId', 'game']].groupby('userId').count().reset_index()
user_freq.columns = ['userId', 'n_ratings']
user_freq.head()

print(f"Mean number of ratings for a given user: {user_freq['n_ratings'].mean():.2f}.")  


import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
ax = sns.countplot(x="rating", data=game, palette="viridis")
plt.title("Distribution of game ratings")

plt.subplot(1,2,2)
ax = sns.kdeplot(user_freq['n_ratings'], shade=True, legend=False)
plt.axvline(user_freq['n_ratings'].mean(), color="k", linestyle="--")
plt.xlabel("# ratings per user")
plt.ylabel("density")
plt.title("Number of games rated per user")
plt.show()

 
# Find Lowest and Highest rated movies:
mean_rating = game.groupby('game')[['rating']].mean()
# Lowest rated movies
lowest_rated = mean_rating['rating'].idxmin()
game.loc[game['userId'] == lowest_rated]
# Highest rated movies
highest_rated = mean_rating['rating'].idxmax()
game.loc[game['userId'] == highest_rated]
# show number of people who rated movies rated movie highest
game[game['userId']==highest_rated]
# show number of people who rated movies rated movie lowest
game[game['userId']==lowest_rated]


## the above movies has very low dataset. We will use bayesian average
game_stats = game.groupby('userId')[['rating']].agg(['count', 'mean'])
game_stats.columns = game_stats.columns.droplevel()
  
# Now, we create user-item matrix using scipy csr matrix
from scipy.sparse import csr_matrix
import numpy as np  
def create_matrix(df):
      
    N = len(df['userId'].unique())
    M = len(df['game'].unique())
      
    # Map Ids to indices
    user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
    game_mapper = dict(zip(np.unique(df["game"]), list(range(M))))
      
    # Map indices to IDs
    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
    game_inv_mapper = dict(zip(list(range(M)), np.unique(df["game"])))
      
    user_index = [user_mapper[i] for i in df['userId']]
    game_index = [game_mapper[i] for i in df['game']]
  
    X = csr_matrix((df["rating"], (game_index, user_index)), shape=(M, N))
      
    return X, user_mapper, game_mapper, user_inv_mapper, game_inv_mapper
  
X, user_mapper, game_mapper, user_inv_mapper, game_inv_mapper = create_matrix(game)
  
from sklearn.neighbors import NearestNeighbors
"""
Find similar movies using KNN
"""
def find_similar_game(game_id, X, k, metric='cosine', show_distance=False):
      
    neighbour_ids = []
      
    game_ind = game_mapper[game_id]
    game_vec = X[game_ind]
    k+=1
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    kNN.fit(X)
    game_vec = game_vec.reshape(1,-1)
    neighbour = kNN.kneighbors(game_vec, return_distance=show_distance)
    for i in range(0,k):
        n = neighbour.item(i)
        neighbour_ids.append(game_inv_mapper[n])
    neighbour_ids.pop(0)
    return neighbour_ids
  
  
game_titles = dict(zip(game['userId'], game['game']))
  
game_id = 1
  
similar_ids = find_similar_game(game_id, X, k=10)
game_title = game_titles[game_id]
  
print(f"Since you watched {game_title}")
for i in similar_ids:
    print(game_titles[i])

