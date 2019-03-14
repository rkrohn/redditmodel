"""
Script that will extract data for a specific subreddit/subreddits for a given year.
- All submissions posted in the given year
- All comments posted in the given year
- All comments posted in the following year/s on posts submitted in the given year.
* this will include `orphaned' comments whose parent comments/submission were posted in the previous year
"""
import pandas as pd
import time
import os
import reddit_mongo_data_utils



if __name__ == "__main__":
    stime = time.ctime()

    # Whether or not to extract comments
    extract_comments = True
    # Whether or not to extract submissions
    extract_submissions = False

    # year to extract data for
    year = 2018
    # subreddit to extract data for
    subreddit = 'changemyview'
    # path to directory to save data in
    save_dir = './reddit_data/{}_{}/'.format(subreddit, year)
    # how many years to collect additional comments for, e.g. 1 - collect additional comments posted in the immediately
    # following year, 0 if no additional comments are to be collected.
    n_years_additional_comments = 0

    # which columns to include for submissions data, if left as [] will include all columns
    columns_to_return_submissions = []
    # which columns to include for comments data, if left as [] will include all columns
    columns_to_return_comments = []


    print('Collecting data sample for {} over {} ... {} '.format(subreddit, year, time.ctime()))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    post_ids = []
    if extract_submissions:
        print("Extracting submissions...")
        # extract and save submissions data for given year
        total_posts = 0
        for monthI in range(1, 2): #13):
            data = reddit_mongo_data_utils.single_subreddit_submissions_data_from_mongo(subreddit, year, monthI,
                                                                                        columns_to_return_submissions,
                                                                                        generate_name_from_id = True)
            post_ids = list(set(post_ids + list(data['name'])))
            total_posts = total_posts + len(data)
            data.to_csv('{}{}_submissions_{}_{}.tsv'.format(save_dir, subreddit, year, monthI), sep='\t',
                        index=False, encoding='utf-8')
            print('   {}_submissions_{}_{}'.format(subreddit, year, monthI), time.ctime())
        print('{} submissions complete ({} collected)\t {}'.format(subreddit, format(total_posts, ','), time.ctime()))

    if extract_comments:
        print("Extracting comments...")
        # extract and save comments data for given year
        total_comments = 0
        for monthI in range(1, 2):
            data = reddit_mongo_data_utils.single_subreddit_comments_data_from_mongo(subreddit, year, monthI,
                                                                                     columns_to_return_comments,
                                                                                     generate_name_from_id = False)
            total_comments = total_comments + len(data)
            data.to_csv('{}{}_comments_{}_{}.tsv'.format(save_dir, subreddit, year, monthI), sep='\t',
                        index=False, encoding='utf-8')
            print('   {}_comments_{}_{}'.format(subreddit, year, monthI), time.ctime())
        etime_comments = time.ctime()
        print('comments complete ({} collected)\t {}'.format(format(total_comments, ','),time.ctime()))

    if n_years_additional_comments > 0:
        print('{} posts to collect additional comments for (if exist)'.format(len(post_ids)))
        print('collecting additional comments...\t {}'.format(time.ctime()))

        total_additional_comments = 0

        year_of_interest = year
        for n_years_after_year_of_interest in range(1,n_years_additional_comments+1):
            year = year_of_interest + n_years_after_year_of_interest

            # extract and save comments data for given year
            for monthI in range(1, 13):
                data = reddit_mongo_data_utils.single_subreddit_comments_data_from_mongo(subreddit, year, monthI,
                                                                                         columns_to_return_comments,
                                                                                         generate_name_from_id=True)
                total_additional_comments = total_additional_comments + len(data)
                save_filepath = '{}{}_{}_additional_comments_{}_{}.tsv'.format(save_dir, subreddit, year_of_interest,
                                                                               year, monthI)
                data.to_csv(save_filepath, sep='\t', index=False, encoding='utf-8')
                print('{}_additional_comments_{}_{}'.format(subreddit, year, monthI), time.ctime())
            etime_comments = time.ctime()
        print('addtional comments complete ({} collected)\t {}'.format(format(total_additional_comments, ','),
                                                                       time.ctime()))

        print('Complete! {} -- {}'.format(stime,time.ctime()))
