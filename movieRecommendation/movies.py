import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
#this function is just to avoid any unnneccessary warnings
warnings.filterwarnings('ignore')
headings=["user_id","item_id","rating","timestamp"]
#using the sep, you can actually read any file as long as
#there is a particular repeatetive seperator
#to add the columns just give the list of the names
main_data=pd.read_csv('ml-100k/u.data', sep='\t',names=headings)
#print(main_data.head())
#. nunique() basically tells you all the unique values
#present in the list
#print(main_data['item_id'].nunique())
movie_titles=pd.read_csv('ml-100k/u.item', sep="\|")
movie_titles=movie_titles.iloc[:,:2]
movie_titles.columns=['item_id','movie_title']
#print(movie_titles.head())
#merge is used to merge to csv files/any files, but they must have a common column
main_data=pd.merge(main_data,movie_titles,on="item_id")
#print(main_data.head(n=150))
#this long function, basically groups the data according to one values
#so the the list is made of movies put together with ther ratings
#then the mean is found according to their ratings
#and then the values are sorted according to their rating values
act_data=main_data.groupby('movie_title').mean()['rating'].sort_values(ascending=False)
#print(act_data.head(n=30))
#the count helps you determine the number of values in the column,
# in this case it determines the number of ratings for a particular movie
next_data=main_data.groupby('movie_title')['rating'].count()
#print(next_data.head(n=20))
#converting both the values to a data frame
ratings=pd.DataFrame(main_data.groupby('movie_title').mean()['rating'])
#print(ratings.head())

#see how to add another column to a DATA frame#RIGHT here
#this format suggests that you group by a particular col
# and then you pass the col for which you want to find the avg
# or the mean or the count etc
ratings['num of ratings']=pd.DataFrame(main_data.groupby('movie_title')['rating'].count())
#print(ratings.head(n=25))
ratings.sort_values(by='rating',ascending=False)
#plt.hist(ratings['rating'],bins=65)
#plt.show()

#sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.7)
#plt.show()
# now we need user-id on one axis and the movie_title on the columns
#and the rating in the cell
rating_pivot_table=main_data.pivot_table(index='user_id',columns='movie_title',values='rating')
#print(rating_pivot_table.head(n=20))
def movie_recommendation(movie_title):
    movie_user_rating=rating_pivot_table[movie_title]
    alike_movie=rating_pivot_table.corrwith(movie_user_rating)
    corr_movie=pd.DataFrame(alike_movie, columns=['Correlation'])
    corr_movie.dropna(inplace=True)
    corr_movie=corr_movie.join(ratings['num of ratings'])
    corr_movie=corr_movie.sort_values(by='Correlation',ascending=False)
    prediction=corr_movie[corr_movie['num of ratings']>100]
    return prediction
predictions=movie_recommendation('Titanic (1997)')
print(predictions.head(n=10))
