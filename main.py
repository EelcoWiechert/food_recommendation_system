from dataloader import *
from rating_matrix import *
from surprise import evaluate, print_perf
from surprise.prediction_algorithms.matrix_factorization import SVD
import numpy as np
from surprise import Reader
from surprise import Dataset

#PARAMETERS
DATA_LOCATION = 'data/u.ratings'  # location where the data is saved
SPLIT_TEST_TRAIN = 0.2          # split test and train dataset, set value to fraction test: max = 1
SAMPLE_SIZE = 0.05               # take a random sample of the users, set value to fraction : max = 1
MIN_RECIPE_RATINGS = 600     # Mimimum number of ratings a recipe should have

#ignore dev by 0 warnings
np.seterr(divide='ignore', invalid='ignore')

# SELECT ALGORITHM
algorithm = 0
# 0 = Content Boosted
# 1 = Hybrid
# 2 = Matrix factorization

'''

Matrix Factorization Algorithm

'''

if algorithm == 2:

    # path to dataset file
    file_path = 'data/u.ratings'

    # As we're loading a custom dataset, we need to define a reader. In the
    # movielens-100k dataset, each line has the following format:
    # 'user item rating timestamp', separated by '\t' characters.
    reader = Reader(line_format='user item rating', sep=',')

    data = Dataset.load_from_file(file_path, reader=reader)
    data.split(n_folds=5)

    algo = SVD()

    # Evaluate performances of our algorithm on the dataset.
    perf = evaluate(algo, data, measures=['RMSE', 'MAE'])

    print(print_perf(perf))

'''

Run the content based and hybrid algorithms

'''

if algorithm == 0 or algorithm == 1:
    #START SCRIPT

    # load the rating matrix
    # this is an matrix with the users as index and recipes as columns
    # the matrix contains the rating
    # it the matrix is not rated, the value is 0
    # it returns also a list with recipes that have been excluded,
    # this is needed to make sure they will not be in the other matrices
    user_recipe_matrix, recipes = load_data(DATA_LOCATION, MIN_RECIPE_RATINGS, SAMPLE_SIZE) # CHECKED

    # a 50% sample is taken from the data
    # the datapoints in the sample are replaced with 0
    # users are the index, columns are the items
    user_recipe_matrix, user_ingredient_matrix_test = split_data(user_recipe_matrix, SPLIT_TEST_TRAIN) # CHECKED

    # the recipe ingredient matrix is created
    # the recipes are the index, the ingredients the columns
    # the value is 1 if the ingredient is in the recipe, 0 otherwise
    ingredient_recipe_matrix = create_ingredient_recipe_matrix(recipes) #CHECKED

    # create the user ingredient matrix
    # this matrix holds the users ratings for ingredients,
    # based on the recipe ratings and the ingredient precense in the recipe
    ingredient_user_matrix = to_ingredients_users_matrix(user_recipe_matrix, ingredient_recipe_matrix) #CHECKED

    # create similarity matrix
    # this is an optional step that can be used to make the user x ingredient matrix less sparse
    # therefore, a collaborative filter is used predicts ingredient ratings based on similar users
    if algorithm == 1:
        ingredient_user_matrix = create_sim_matrix(user_recipe_matrix, ingredient_user_matrix)

    # create the user x recipe matrix based on the ingredient ratings
    # this matrix contains the ratings for the recipes
    predictions = to_recipes_users_matrix(ingredient_user_matrix, ingredient_recipe_matrix)

    # measure the accuracy of the predictions
    # for each user in the test dataset, the difference is used to
    RMSE(user_ingredient_matrix_test, predictions)
    MSE(user_ingredient_matrix_test, predictions)
    RMSE_alternative(user_ingredient_matrix_test, predictions)
    MSE_alternative(user_ingredient_matrix_test, predictions)