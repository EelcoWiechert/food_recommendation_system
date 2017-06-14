import csv
import json
from evaluation import *

def load_data(file, limit, SAMPLE_SIZE): # CHECKED
    print('')
    print('=================')
    print('STEP 1 of 7 - Loading data...')
    print('=================')
    print('')
    if True:
        print('Writing original review file to u.ratings')
        counter=0
        user_reviews_number={}
        item_reviews_number={}
        with open(file, 'w') as rating_file:
            writer = csv.writer(rating_file, delimiter=',')
            for line in open('data/reviews.data'):
                counter += 1
                # count number of reviews for users and items
                review = json.loads(line.strip())
                if int(review['recipe']) in item_reviews_number.keys():
                    item_reviews_number[int(review['recipe'])] +=1
                else:
                    item_reviews_number[int(review['recipe'])] = 1

                if int(review['user']) in user_reviews_number.keys():
                    user_reviews_number[int(review['user'])] +=1
                else:
                    user_reviews_number[int(review['user'])] = 1

            # set limit = minimum number of reviews in order for the recipe
            # to be taken into account
            recipes={}
            for line in open('data/reviews.data'):
                review = json.loads(line)
                if item_reviews_number[int(review['recipe'])] <limit:
                    continue
                else:
                    recipes[int(review['recipe'])] = 0
                    to_write = [int(review['user']), int(review['recipe']), int(review['rating'])]
                    writer.writerow(to_write)

    print('Create User-Recipe Matrix')

    ratings = pd.read_csv(file, sep=',', names=['user_id', 'recipe_id', 'rating'], encoding='latin-1')
    ratings = ratings.sort_values(['recipe_id', 'user_id'], ascending=[True, True])
    recipe_user_matrix = ratings.pivot_table(index='user_id', columns='recipe_id', values='rating')
    recipe_user_matrix = recipe_user_matrix.fillna(0.0)  # Fill non rated items with 0
    recipe_user_matrix = recipe_user_matrix.sample(frac=SAMPLE_SIZE)
    print('Shape user x recipe:')
    print(recipe_user_matrix.shape)
    sparcity_numpy(recipe_user_matrix.as_matrix(), "user_recipe_matrix")
    return recipe_user_matrix, recipes

def split_data(df, fraction): #checked
    print('')
    print('=================')
    print('STEP 2 of 7 - Creating a sample...')
    print('=================')
    print('')
    test_data = df.sample(frac=fraction).sample(frac=fraction, axis=1)
    print('Percentage users included in train set: %s' %(fraction))
    print('Percentage users included in test set: %s' %(fraction))

    # set test data to 0 in original dataframe
    recipes_test = test_data.columns.values
    users_test = test_data.index.values
    for user in users_test:
        for recipe in recipes_test:
            df.set_value(user, recipe, 0)
    return df, test_data