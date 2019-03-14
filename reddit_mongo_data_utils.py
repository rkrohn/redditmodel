"""
Functions that provide basic functionality to extract data from reddit mongo databases.
Last updated by Maria, Feb 2019
"""
from pymongo import MongoClient
import pandas as pd

ALL_REDDIT_MONGO_COMMENT_COLUMNS = []
ALL_REDDIT_MONGO_SUBMISSIONS_COLUMNS = []

REDDIT_PREFIX = {'comments':'t1_','submissions':'t3_','subreddit':'t5_'}

"""
Functions to pull user data from mongo
"""

def generate_protection_query_dict(columns_to_return):
    """
    Generates the projections query for specified columns to return.
    If columns_to_return is an empty list ([]), then query is {} and all columns are returned.
    """
    projection_query = {}
    for col in columns_to_return:
        projection_query[col] = 1
    return projection_query


def extract_data(selection_query, projection_query, collection):
    """
    Runs find query for given selection_query and projection_query on collection,
    creates dataframe from resulting cursor (iterator with json objects that matched selection_query).
    If no objects matched, an empty dataframe is created with the columns specified in projection_query.
    If no objects matched and {} is used for projection_query, empty dataframe created if no matches will have no columns.
    """
    if len(projection_query.keys()) == 0:
        data = collection.find(selection_query)
    else:
        data = collection.find(selection_query, projection_query)
    data = list(data)
    if len(data) > 0:
        data = pd.DataFrame(data)
    else:
        data = pd.DataFrame(columns=projection_query.keys())
    return data


def get_collection_suffix(year, month):
    """
    Returns yyyy_mm string for year and month passed.
    """
    collection_suffix = '{}_{}'.format(year, '0{}'.format(month)[-2:])
    return collection_suffix


def get_count_where_match_against_value(key_to_match_against, value_to_match, year, month, columns_to_return, object_type):
    """
    Returns count of objects of given object_type ('submissions' or 'comments') where key_to_match_against is the value
    passed in value_to_match from the collection for the given year and month.
    Returns count as int.
    """
    database_name = 'reddit{}'.format(year)

    collection_suffix = get_collection_suffix(year, month)
    print('{}.{}_{}'.format(database_name, object_type, collection_suffix))

    selection_query = {key_to_match_against: value_to_match}
    projection_query = generate_protection_query_dict(columns_to_return)

    collection = MongoClient()[database_name]['{}_{}'.format(object_type, collection_suffix)]
    n = collection.find(selection_query, projection_query).count()
    return n

def get_count_where_match_against_list(key_to_match_against, list_of_values_to_match, year, month,
                                       columns_to_return, object_type):
    """
    Returns count of objects of given object_type ('submissions' or 'comments') where key_to_match_against is the value
    passed in value_to_match from the collection for the given year and month.
    Returns count as int.
    """
    database_name = 'reddit{}'.format(year)

    collection_suffix = get_collection_suffix(year, month)
    print('{}.{}_{}'.format(database_name, object_type, collection_suffix))

    selection_query = {key_to_match_against : {"$in" : list_of_values_to_match}}
    projection_query = generate_protection_query_dict(columns_to_return)

    collection = MongoClient()[database_name]['{}_{}'.format(object_type, collection_suffix)]
    n = collection.find(selection_query, projection_query).count()
    return n

def get_data_from_mongo_where_match_against_value(key_to_match_against, value_to_match,
                                                 year, month, columns_to_return, object_type,
                                                 generate_name_from_id=True):
    """
    Pulls object_type ('submissions' or 'comments') where key_to_match_against is the value
    passed in value_to_match from the collection for the given year and month.
    Returns a pandas dataframe, data, with columns specified in columns_to_return.
    e.g.
    submissions posted by users in list_of_users:
     key_to_match_against = 'author'
     list_of_values_to_match = list_of_users
    submissions posted to subreddits in list_of_subreddits:
     key_to_match_against = 'subreddit'
     list_of_values_to_match = list_of_subreddits
    """
    database_name = 'reddit{}'.format(year)

    collection_suffix = get_collection_suffix(year, month)
    #print('{}.{}_{}'.format(database_name, object_type, collection_suffix))

    selection_query = {key_to_match_against: value_to_match}
    projection_query = generate_protection_query_dict(columns_to_return)

    collection = MongoClient()[database_name]['{}_{}'.format(object_type, collection_suffix)]
    data = extract_data(selection_query, projection_query, collection)
    if generate_name_from_id:
        if len(data) > 0:
            ## sometimes the name column is corrupted
            ## to avoid data cleaning later, simply create name column based off id
            data['name'] = ['{}_{}'.format(REDDIT_PREFIX[object_type], x) for x in data['id']]
        else:
            data = pd.DataFrame(columns = columns_to_return + ['id'])
    return data

def get_data_from_mongo_where_match_against_list(key_to_match_against, list_of_values_to_match,
                                                 year, month, columns_to_return, object_type,
                                                 generate_name_from_id = True):
    """
    Pulls object_type ('submissions' or 'comments') where key_to_match_against is one of the values
    passed in list_of_values_to_match from the collection for the given year and month.
    Returns a pandas dataframe, data, with columns specified in columns_to_return.
    e.g.
    submissions posted by users in list_of_users:
     key_to_match_against = 'author'
     list_of_values_to_match = list_of_users
    submissions posted to subreddits in list_of_subreddits:
     key_to_match_against = 'subreddit'
     list_of_values_to_match = list_of_subreddits
    """
    database_name = 'reddit{}'.format(year)

    collection_suffix = get_collection_suffix(year, month)
    print('{}.{}_{}'.format(database_name, object_type, collection_suffix))

    selection_query = {key_to_match_against : {"$in" : list_of_values_to_match}}
    projection_query = generate_protection_query_dict(columns_to_return)

    collection = MongoClient()[database_name]['{}_{}'.format(object_type, collection_suffix)]
    data = extract_data(selection_query, projection_query, collection)
    if generate_name_from_id:
        if len(data) > 0:
            ## sometimes the name column is corrupted
            ## to avoid data cleaning later, simply create name column based off id
            data['name'] = ['{}_{}'.format(REDDIT_PREFIX[object_type], x) for x in data['id']]
        else:
            data = pd.DataFrame(columns=columns_to_return+['name'])
    return data


def single_user_comments_data_from_mongo(user, year, month, columns_to_return, generate_name_from_id = True):
    """
    Pulls comments posted by user from the collection for the given year and month
    and returns a pandas dataframe, data, with columns specified in columns_to_return.
    """
    return get_data_from_mongo_where_match_against_value('author', user,
                                                  year, month, columns_to_return, 'comments',
                                                  generate_name_from_id=generate_name_from_id)


def multiple_users_comments_data_from_mongo(list_of_users, year, month, columns_to_return, generate_name_from_id = True):
    """
    Pulls comments posted by users in list_of_users from the collection for the given year and month
    and returns a pandas dataframe, data, with columns specified in columns_to_return.
    """
    return get_data_from_mongo_where_match_against_list('author', list_of_users,
                                                 year, month, columns_to_return, 'comments',
                                                 generate_name_from_id=generate_name_from_id)


def single_user_submissions_data_from_mongo(user, year, month, columns_to_return, generate_name_from_id = True):
    """
    Pulls submissions posted by user from the collection for the given year and month
    and returns a pandas dataframe, data, with columns specified in columns_to_return.
    """
    return get_data_from_mongo_where_match_against_value('author', user,
                                                  year, month, columns_to_return, 'submissions',
                                                  generate_name_from_id=generate_name_from_id)


def multiple_users_submissions_data_from_mongo(list_of_users, year, month, columns_to_return,
                                               generate_name_from_id = True):
    """
    Pulls submissions posted by users in list_of_users from the collection for the given year and month
    and returns a pandas dataframe, data, with columns specified in columns_to_return.
    """
    return get_data_from_mongo_where_match_against_list('author', list_of_users,
                                                 year, month, columns_to_return, 'submissions',
                                                 generate_name_from_id=generate_name_from_id)



def single_subreddit_comments_data_from_mongo(subreddit, year, month, columns_to_return, generate_name_from_id = True):
    """
    Pulls comments posted by user from the collection for the given year and month
    and returns a pandas dataframe, data, with columns specified in columns_to_return.
    """
    return get_data_from_mongo_where_match_against_value('subreddit', subreddit,
                                                  year, month, columns_to_return, 'comments',
                                                  generate_name_from_id=generate_name_from_id)


def multiple_subreddit_comments_data_from_mongo(list_of_subreddits, year, month, columns_to_return, generate_name_from_id = True):
    """
    Pulls comments posted by users in list_of_users from the collection for the given year and month
    and returns a pandas dataframe, data, with columns specified in columns_to_return.
    """
    return get_data_from_mongo_where_match_against_list('subreddit', list_of_subreddits,
                                                 year, month, columns_to_return, 'comments',
                                                 generate_name_from_id=generate_name_from_id)

def single_subreddit_submissions_data_from_mongo(subreddit, year, month, columns_to_return, generate_name_from_id = True):
    """
    Pulls submissions posted by user from the collection for the given year and month
    and returns a pandas dataframe, data, with columns specified in columns_to_return.
    """
    return get_data_from_mongo_where_match_against_value('subreddit', subreddit,
                                                  year, month, columns_to_return, 'submissions',
                                                  generate_name_from_id=generate_name_from_id)


def multiple_subreddit_submissions_data_from_mongo(list_of_subreddits, year, month, columns_to_return,
                                               generate_name_from_id = True):
    """
    Pulls submissions posted by users in list_of_users from the collection for the given year and month
    and returns a pandas dataframe, data, with columns specified in columns_to_return.
    """
    return get_data_from_mongo_where_match_against_list('subreddit', list_of_subreddits,
                                                 year, month, columns_to_return, 'submissions',
generate_name_from_id=generate_name_from_id)