import json
from collections import defaultdict


def count_user_item_ratings(data_full):
    """
    Count the number of ratings for each user and item
    """
    countU = defaultdict(lambda: 0)
    countP = defaultdict(lambda: 0)
    
    for index, row in data_full.iterrows():
        user_id = row['user_id']
        parent_asin = row['parent_asin']
        countU[user_id] += 1
        countP[parent_asin] += 1
        
    return countU, countP

def map_and_filter_data(data_full, countU, countP, args):
    """
    Map users and items to integer IDs and filter out those with fewer than min_ratings
    """
    usermap = dict()
    usernum = 0
    itemmap = dict()
    itemnum = 0
    User = dict()
    num_interactions = 0
    for index, row in data_full.iterrows():
        user_id = row['user_id']
        parent_asin = row['parent_asin']
        timestamp = row['timestamp']
        num_interactions += 1
        if countU[user_id] < args.least_interaction or countP[parent_asin] < args.least_interaction:
            continue
            
        if user_id in usermap:
            userid = usermap[user_id]
        else:
            usernum += 1
            userid = usernum
            usermap[user_id] = userid
            User[userid] = []
            
        if parent_asin in itemmap:
            itemid = itemmap[parent_asin]
        else:
            itemnum += 1
            itemid = itemnum
            itemmap[parent_asin] = itemid
            
        User[userid].append([timestamp, itemid])
    
    dataset_stats = {
                    "num_users": usernum,
                    "num_items": itemnum, 
                    "num_interactions": num_interactions
                     }
    with open(f"dataset/{args.categories[0]}_stats.json", "w") as f:
        json.dump(dataset_stats, f, indent=4)
    
    return usermap, usernum, itemmap, itemnum, User

def sort_user_data(User):
    """
    Sort user ratings by timestamp
    """
    for userid in User.keys():
        User[userid].sort(key=lambda x: x[0])
    return User

def write_output_file(User, args):
    """
    Write processed data to output file
    """
    with open(f'dataset/{args.categories[0]}.txt', 'w') as f:
        for user in User.keys():
            for i in User[user]:
                f.write('%d %d\n' % (user, i[1]))
    print(f"Data written to {f'dataset/{args.categories[0]}'}.txt")

def pre_process_data(data_full, args):
    """
    Pre-processes the raw input data by:
    1. Counting user-item ratings
    2. Mapping users and items to internal IDs and filtering data
    3. Sorting user data by timestamp
    4. Writing the processed data to output files
    Parameters:
    ----------
    data_full : DataFrame or similar
        Raw input data containing user-item interactions
    arg : argparse.Namespace
        Arguments containing processing parameters including categories
    Returns:
    -------
    None
        Results are written to files specified by arg.categories
    """

    countU, countP = count_user_item_ratings(data_full)
    
    usermap, usernum, itemmap, itemnum, User = map_and_filter_data(data_full, countU, countP, args=args)
    
    User = sort_user_data(User)
    
    print(usernum, itemnum)
    
    write_output_file(User, args)
    
    key = 2
    res = {k: v for i, (k, v) in enumerate(itemmap.items()) if i < key}
    print(res)

    print("Data pre-processing completed.")

    return usermap, usernum, itemmap, itemnum