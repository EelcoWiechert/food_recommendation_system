import json
from evaluation import *
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

# create ingredient matrix
def create_ingredient_recipe_matrix(included_recipes):
    print('')
    print('=================')
    print('STEP 3 of 7 - Loading recipe x ingredient  matrix')
    print('=================')
    print('')
    with open('data/recipe_ingredient_library.json') as json_data:
        d = json.loads(json_data.read())

    ingredients = []
    recipes = []
    for r, i in d.items():
        if int(r) not in included_recipes:
            continue
        else:
            recipes.append(int(r))
            for item in i:
                ingredients.append(item)

    ingredients = set(ingredients)
    ingredients = sorted(ingredients, reverse=False)
    recipes = sorted(recipes, key=int, reverse=False)

    final_array = []
    count = 0
    for recipe in recipes:
        count += 1
        temp = []
        for ing in ingredients:
            if ing in d[str(recipe)]:
                temp.append(1.0)
            else:
                temp.append(0.0)
        final_array.append(temp)

    final_array = np.array(final_array)

    ingredient_recipe_matrix = pd.DataFrame(data=final_array, columns=ingredients, index=recipes)
    print(ingredient_recipe_matrix.shape)
    print('Ingredient-Recipe Matrix loaded')

    return ingredient_recipe_matrix


# multiply rating matrix with ingredient matrix
def to_ingredients_users_matrix(recipe_user_matrix, ingredient_recipe_matrix):

    print('')
    print('==============================')
    print('STEP 4 of 7 - Create user ingredient matrix')
    print('==============================')
    print('')

    URM = np.where(recipe_user_matrix.as_matrix() > 0,1,0)
    dataframe = pd.DataFrame(data=URM, columns=list(recipe_user_matrix.columns), index=list(recipe_user_matrix.index))

    # multiply recipe-user (0-1) with ingredient-recipe
    division_matrix = dataframe.dot(ingredient_recipe_matrix)

    ingredient_user_matrix = (recipe_user_matrix.dot(ingredient_recipe_matrix)).divide(division_matrix).round(1).fillna(0.0)
    print('Ingredient User matrix loaded')
    print(ingredient_user_matrix.shape)
    sparcity_numpy(ingredient_user_matrix.as_matrix(), "user_ingredient_matrix")
    return ingredient_user_matrix

def to_recipes_users_matrix(user_ingredient_matrix, ingredient_recipe_matrix):
    print('')
    print('=========================')
    print('STEP 6 of 7 - Create recipe user matrix')
    print('=========================')
    print('')

    dataframe = user_ingredient_matrix.dot(ingredient_recipe_matrix.T)
    URM = np.where(user_ingredient_matrix.as_matrix() > 0, 1, 0)
    one_zero_matrix = pd.DataFrame(data=URM, columns=list(user_ingredient_matrix.columns), index=list(user_ingredient_matrix.index))

    # create an user x ingredient matrix that only contains ones and zeros
    # 1 if there is a rating of the user for the ingredient
    # 0 otherwise
    temp = one_zero_matrix.dot(ingredient_recipe_matrix.T)
    dataframe2 = dataframe.divide(temp).round(1).fillna(0.0)
    print('recipe user matrix loaded')
    print(dataframe.shape)
    sparcity_numpy(dataframe2.as_matrix(), 'results')
    return dataframe2


# CREATE SIMILARITY MATRIX
# THIS IS A MATRIX USERS-USER MATRIX THAT CONTAINS FOR EACH USER HOW SIMILAR IT IS TO THE OTHER USER

def create_sim_matrix(user_recipe_matrix, user_ingredient_matrix):
    print('')
    print('==============================')
    print('STEP 5 of 7 - Create similarity matrix')
    print('==============================')
    print('')

    # --- Start User Based Recommendations --- #

    def predict(ratings, similarity):
        ratings_one_zero = np.where(ratings > 0, 1, 0)
        return similarity.dot(ratings) / similarity.dot(ratings_one_zero)

    # Create a place holder matrix for similarities
    A_sparse = sparse.csr_matrix(user_recipe_matrix.as_matrix())
    similarities = cosine_similarity(A_sparse)
    print('pairwise dense output:\n {}\n'.format(similarities))

    user_ingredient_array = predict(user_ingredient_matrix.as_matrix(), similarities)
    sparcity_numpy(np.where(user_ingredient_array > 0.01, 1, 0), 'user_ingredient_after_CF')
    user_ingredient_matrix = pd.DataFrame(data=user_ingredient_array, index= list(user_ingredient_matrix.index),columns=list(user_ingredient_matrix.columns)).fillna(0.0)
    print(user_ingredient_matrix.head())
    print('Number of users: %s' %(str(user_recipe_matrix.shape[0])))
    print('Number of recipes: %s' %(str(user_recipe_matrix.shape[1])))

    return user_ingredient_matrix

def create_division_matrix_similarities(data):
    count = 0
    division_matrix = data
    for user in list(data.index):
        for ingredient in list(data.columns):
            if data.loc[user][ingredient] > 0:
                division_matrix.loc[user][ingredient] = 1

            else:
                division_matrix.loc[user][ingredient] = 0
        count += 1
        if count %100 == 0:
            print(count)

    return division_matrix

def matrix_one_zero(matrix):
    count = 0
    for row in list(matrix.index):
        for column in list(matrix.columns):
            if matrix.loc[row][column] > 0:
                matrix.loc[row][column] = 1
            else:
                matrix.loc[row][column] = 0
        count += 1
        if count %100 == 0:
            print(count)

    return matrix

def person_correlation(person1, person2):
    # To get both rated items

    # get average of both users
    try:
        avg_person1 = np.mean([x for x in person1 if x != 0])
        avg_person2 = np.mean([x for x in person2 if x != 0])
    except:
        return 0

    nominator = 0
    denominator_person1 = 0
    denominator_person2 = 0

    error=0
    for value in range(len(person1)):
        if person1[value] != 0 and person2[value] != 0: #if both users rated the same item
            nominator += ((person1[value] - avg_person1) * (person2[value] - avg_person2))
            denominator_person1 += math.pow((person1[value] - avg_person1),2)
            denominator_person2 += math.pow((person2[value] - avg_person2), 2)
        else:
            error+=1

    if error == len(person1) or (person1[value] - avg_person1) == 0 or (person2[value] - avg_person2) == 0 :
        return 0

    return (nominator / (denominator_person1*denominator_person2))