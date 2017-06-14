import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

def rating_distribution(rating_matrix):
    print('')
    print('====================')
    print('create rating statistics and figures')
    print('====================')

    ratings = pd.read_csv(file, sep=',', names=['user_id', 'recipe_id', 'rating'], encoding='latin-1')

    print('There are %s user included in the dataset' % (rating_matrix.shape[0]))


    ratings = rating_matrix['rating'].tolist()

    print('Mean of the average recipe rating is %s and standard dev is %s, skewness: %s' % (
        np.mean(ratings, axis=0), np.std(ratings, axis=0), stats.skew(ratings, axis=0)))

    average_rating = float(sum(ratings)) / float(len(ratings))

    # the histogram of the data
    plt.hist(ratings, bins=5, edgecolor='black', range=[0, 5])
    plt.axvline(x=average_rating, color='r', linestyle='dashed', linewidth=2,
                label='Average rating : ' + str(round(average_rating, 2)))
    plt.legend()

    plt.title("Distribution of ratings (recipe independent)")
    plt.xlabel("Review score")
    plt.ylabel("Frequency")

    plt.savefig('/Users/eelcowiechert/Documents/A TUE/2016-2017 Q3 & Q4/Project/Literature '
                'study/av_recipe_rating_users.png')
    plt.close()

    # Average rating per recipe


    print('')
    print(rating_matrix.groupby(['recipe_id'])['rating'].mean().head())

    ratings = []
    for index, row in rating_matrix.groupby(['recipe_id'])['rating'].mean().iteritems():
        ratings.append(round(row,2))

    print('Mean of the average recipe rating is %s and standard dev is %s, skewness: %s' % (
    np.mean(ratings, axis=0), np.std(ratings, axis=0), stats.skew(ratings, axis=0)))

    average_rating = float(sum(ratings)) / float(len(ratings))

    # the histogram of the data
    plt.hist(ratings, bins=40, edgecolor='black', range=[0, 5])
    plt.axvline(x=average_rating, color='r', linestyle='dashed', linewidth=2,
                label='Average rating : ' + str(round(average_rating, 2)))
    plt.legend()

    plt.title("Recipe rating (average rating per recipe)")
    plt.xlabel("Rating from 1 to 5 stars")
    plt.ylabel("Frequency")

    plt.savefig('/Users/eelcowiechert/Documents/A TUE/2016-2017 Q3 & Q4/Project/Literature study/av_recipe_rating.png')
    plt.close()

    # Number of ratings per recipe
    ratings =[]
    excluded = 0
    for t in rating_matrix.recipe_id.value_counts().iteritems():
        if t[1] > 3:
            ratings.append(t[1])
        else:
            excluded +=1
    print('')
    print('There are %s recipes excluded because they have not enough recipes rated' % excluded)

    average_rating = float(sum(ratings)) / float(len(ratings))
    min_rating = min(ratings)
    max_rating = max(ratings)

    plt.axvline(x=average_rating, color='r', linestyle='dashed', linewidth=0.5,
                label='average # of ratings : ' + str(int(average_rating)))
    plt.axvline(x=min_rating, color='blue', linestyle='dotted', linewidth=0.5,
                label='minimum / maximum # of ratings : ' + str(int(min_rating)) + ' / ' + str(int(max_rating)))
    plt.axvline(x=max_rating, color='blue', linestyle='dotted', linewidth=0.5)
    plt.hist(ratings, bins=75, edgecolor='black')
    plt.yscale('log')
    plt.legend()
    plt.title("Quantity of reviews per recipe (distribution)")
    plt.xlabel("# of ratings")
    plt.ylabel("Frequency")
    plt.savefig('/Users/eelcowiechert/Documents/A TUE/2016-2017 Q3 & Q4/Project/Literature study/number_of_ratings.png')
    plt.close()

    # Number of reviews per user
    ratings =[]
    excluded = 0
    for c in rating_matrix.user_id.value_counts().iteritems():
        if c[1] > 5:
            ratings.append(c[1])
        else:
            excluded += 1

    print('There are %s user excluded because they have not enough recipes rated' % excluded)
    print('')
    average_rating = float(sum(ratings)) / float(len(ratings))
    min_rating = min(ratings)
    max_rating = max(ratings)

    plt.axvline(x=average_rating, color='r', linestyle='dashed', linewidth=0.5,
                label='average # of ratings : ' + str(int(average_rating)))
    plt.axvline(x=min_rating, color='blue', linestyle='dotted', linewidth=0.5,
                label='minimum / maximum # of ratings : ' + str(int(min_rating)) + ' / ' + str(int(max_rating)))
    plt.axvline(x=max_rating, color='blue', linestyle='dotted', linewidth=0.5)
    plt.hist(ratings, bins=75, edgecolor='black')
    plt.legend()
    plt.yscale('log')
    plt.title("Distribution of quality of user reviews")
    plt.xlabel("# of ratings")
    plt.ylabel("Frequency")
    plt.savefig('/Users/eelcowiechert/Documents/A TUE/2016-2017 Q3 & Q4/Project/Literature study/number_of_reviews.png')
    plt.close()

def RMSE(actual, predict):
    print('')
    print('=========================')
    print('STEP 7 of 7 - Evaluation')
    print('=========================')
    print('')
    actual_list =[]
    predict_list =[]
    for user in list(actual.index):
        for ingredient in list(actual.columns):
            if actual.loc[user][ingredient] == 0:# or predict.loc[user][ingredient] == 0:
                continue
            else:
                actual_list.append(actual.loc[user][ingredient])
                predict_list.append(predict.loc[user][ingredient])

    count = 0
    number_of_ratings = 0
    difference = 0
    for rating in actual_list:
        if rating > 0:
            dif = math.pow((actual_list[count] - predict_list[count]), 2)
            difference += dif
            number_of_ratings +=1
        count += 1

    print('RMSE: %s' % (math.sqrt(difference / number_of_ratings)))
    return math.sqrt(difference / number_of_ratings)

def RMSE_alternative(actual, predict):
    print('')
    print('=========================')
    print('STEP 7 of 7 - Evaluation')
    print('=========================')
    print('')
    actual_list =[]
    predict_list =[]
    for user in list(actual.index):
        for ingredient in list(actual.columns):
            if actual.loc[user][ingredient] == 0 or predict.loc[user][ingredient] < 0.1:
                continue
            else:
                actual_list.append(actual.loc[user][ingredient])
                predict_list.append(predict.loc[user][ingredient])

    count = 0
    number_of_ratings = 0
    difference = 0
    for rating in actual_list:
        if rating > 0:
            dif = math.pow((actual_list[count] - predict_list[count]), 2)
            difference += dif
            number_of_ratings +=1
        count += 1

    print('RMSE_alternative: %s' % (math.sqrt(difference / number_of_ratings)))
    return math.sqrt(difference / number_of_ratings)

def MSE_alternative(actual, predict):
    actual_list =[]
    predict_list =[]
    for user in list(actual.index):
        for ingredient in list(actual.columns):
            if actual.loc[user][ingredient] == 0 or predict.loc[user][ingredient] < 0.1:
                continue
            else:
                actual_list.append(actual.loc[user][ingredient])
                predict_list.append(predict.loc[user][ingredient])

    print('MAE_alternative: %s' % (mean_absolute_error(actual_list,predict_list)))
    print('MSE_alternative: %s' % (mean_squared_error(actual_list, predict_list)))
    #return (difference / number_of_ratings)

def MSE(actual, predict):
    actual_list =[]
    predict_list =[]
    for user in list(actual.index):
        for ingredient in list(actual.columns):
            if actual.loc[user][ingredient] == 0:# or predict.loc[user][ingredient] == 0:
                continue
            else:
                actual_list.append(actual.loc[user][ingredient])
                predict_list.append(predict.loc[user][ingredient])

    print('MAE: %s' % (mean_absolute_error(actual_list,predict_list)))
    print('MSE: %s' % (mean_squared_error(actual_list, predict_list)))
    #return (difference / number_of_ratings)

def sparcity_numpy(numpy_matrix, name):
    sparsity = float(len(numpy_matrix.nonzero()[0]))
    sparsity /= (numpy_matrix.shape[0] * numpy_matrix.shape[1])
    sparsity *= 100
    print('Sparsity ' + name + ': {:4.2f}%'.format(sparsity))