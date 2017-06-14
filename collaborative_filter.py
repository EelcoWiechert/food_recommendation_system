# import code
from math import sqrt
import numpy as np
import math


def similarity_score(person1, person2):
    # this Returns the ration euclidean distancen score of person 1 and 2

    # To get both rated items by person 1 and 2
    both_viewed = {}

    for item in dataset[person1]:
        if item in dataset[person2]:
            both_viewed[item] = 1

        # The Conditions to check if they both have common rating items
        if len(both_viewed) == 0:
            return 0

        # Finding Euclidean distance
        sum_of_eclidean_distance = []

        for item in dataset[person1]:
            if item in dataset[person2]:
                sum_of_eclidean_distance.append(pow(dataset[person1][item] - dataset[person2][item], 2))
        sum_of_eclidean_distance = sums(sum_of_eclidean_distance)

        return 1 / (1 + sqrt(sum_of_eclidean_distance))


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

    if error == len(person1):
        return 0

    return (nominator / (denominator_person1*denominator_person2))


def most_similar_users(person, number_of_users):
    # returns the number_of_users (similar persons) for a given specific person
    scores = [(person_correlation(person, other_person), other_person) for other_person in dataset if
              other_person != person]

    # Sort the similar persons so the highest scores person will appear at the first
    scores.sort()
    scores.reverse()
    return scores[0:number_of_users]


def user_recommendations(person, dataset):
    # Gets recommendations for a person by using a weighted average of every other user's rankings
    totals = {}
    simSums = {}
    rankings_list = []
    for other in list(dataset.index.values):
        # don't compare me to myself
        if other == person:
            continue
        sim = person_correlation(person, other)
        # print ">>>>>>>",sim

        # ignore scores of zero or lower
        if sim <= 0:
            continue
        for item in dataset[other]:

            # only score movies i haven't seen yet
            if item not in dataset[person] or dataset[person][item] == 0:
                # Similrity * score
                totals.setdefault(item, 0)
                totals[item] += dataset[other][item] * sim
                # sum of similarities
                simSums.setdefault(item, 0)
                simSums[item] += sim

                # Create the normalized list

    rankings = [(total / simSums[item], item) for item, total in totals.items()]
    rankings.sort()
    rankings.reverse()
    # returns the recommended items
    recommendataions_list = [recommend_item for score, recommend_item in rankings]
    return recommendataions_list


print
user_recommendations('Alice')