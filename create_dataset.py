import os
import sys
import boto3
import gzip
import pandas as pd
from operator import itemgetter
from tabulate import tabulate

S3BUCKET = 'amazon-reviews-pds'


def connect_to_s3bucket(bucket_name: str = None):
    # Using boto3's resource method, create a connection to AWS S3 and extract from bucket amazon-reviews-pds:
    s3conn = boto3.resource(
        's3',
        # If you have an AWS account, replace these with your key-id and access key:
        aws_access_key_id=os.environ['AKID'],
        aws_secret_access_key=os.environ['SAK']
    )
    bucket = s3conn.Bucket(bucket_name)
    return bucket


def create_data(bucket, keys_list_indices: int = 11, key='tmp.gz'):
    # Filter the keys to include only tsv objects containing reviews, and only from the US
    keys_list = []
    for my_bucket_object in bucket.objects.all():
        keys_list.append([my_bucket_object.key, my_bucket_object.size])
    keys_list_filtered = list(filter(lambda l: "tsv.gz" in l[0] and "us" in l[0], keys_list))
    fileToStream = keys_list_filtered[keys_list_indices][0]  # keys_list[13][0]

    # Reading the file (may take time?)
    bucket.download_file(fileToStream, key)
    with gzip.open(key, 'rb') as f_in:
        tmp = [x.decode('utf8').strip() for x in
               f_in.readlines()]  # f_in.readlines() # Reading lines into a python object

    # Read tmp  into a pandas dataframe
    df = pd.DataFrame([tmp[i].split('\t') for i in range(1, len(tmp))], columns=tmp[0].split('\t'))
    return keys_list_filtered, df


def main():
    reviews_bucket = connect_to_s3bucket(bucket_name=S3BUCKET)
    keys_list_filtered, _ = create_data(bucket=reviews_bucket, keys_list_indices=0)
    # data for printing after converting size from bytes to GB,rounded to 3 dec.
    print_objectskeys = list(map(lambda l: [l[0], round(l[1] / (1024 ** 3), 3)], keys_list_filtered))
    print(tabulate(sorted([print_objectskeys[i] for i in range(15)], key=itemgetter(1), reverse=True),
                   headers=['idx', 'Keys[tsv]', 'Size[in GB]'],
                   tablefmt='fancy_grid', showindex="always"))

    category = []
    # save the file's (object's) name in a list and count how many times they appeared in this list
    word = [keys_list_filtered[i][0].split('us_')[len(keys_list_filtered[i][0].split('us_')) - 1].split('_v1')[0] for i
            in range(len(keys_list_filtered))]
    # create a dictionary of the the object name and the number of time it appeared in the list
    word_count = dict((i, word.count(i)) for i in word)
    # iter over the categories list
    for i in range(len(keys_list_filtered)):
        if word_count[word[i]] > 1:  # add a suffix only if the word appeared more than once
            # retrieve the suffix from the original file's name and add it to the name
            suffix = \
                keys_list_filtered[i][0].split('_v1')[len(keys_list_filtered[i][0].split('_v1')) - 1].split('.tsv')[0]
            category.append(word[i] + suffix)
        else:  # else leave it as it is
            category.append(word[i])

    # create lists of sizes
    size = list(map(itemgetter(1), keys_list_filtered))
    sizeGB = list(map(lambda l: l / (1024 ** 3), size))
    estSizeGB = list(map(lambda l: l * 3.33, sizeGB))
    # sizeGB = round(sizeGB ,3)

    # add the lists to a data frame: file_categories_df
    file_categories_df = pd.DataFrame({'category': category,
                                       'size': size,
                                       'sizeGB': sizeGB,
                                       'estSizeGB': estSizeGB})

    # Show the top-10 rows sorted alphabetically
    print(tabulate(file_categories_df.head(10), headers='keys', tablefmt='fancy_grid',
                   showindex="always"))

    # check what categories are there with size < 30MB=0.03 GB
    cat_under_30MB = file_categories_df.loc[file_categories_df['sizeGB'] < 0.03, 'category']

    print('file to read/stream: ', keys_list_filtered[11][0])  # downloading: Major_Appliances.

    _, df = create_data(bucket=reviews_bucket, keys_list_indices=cat_under_30MB.index[0])
    print("Digital_Software has {} rows(data points) and {} columns(features)".format(df.shape[0], df.shape[1]))

    # Compute and print the average size in bytes of each data point
    avg_size = (sum([sys.getsizeof(v) for v in df.values]) + df.values.nbytes) / len(df)
    print("The average size[bytes] of each data point (an amazon product review) is: {}".format(avg_size))

    # estimate the total number of reviews in the entire dataset over all categories
    est_num_rev = sum([file_categories_df['size'][i] / avg_size for i in
                       range(len(file_categories_df))])  # I took the each category size and divided it
    # by the average size of the df data points[all in bytes]
    print("The estimated total number of reviews in the entire dataset over all categories is: {}".format(
        round(est_num_rev, 3)))

    df.columns = list(df.columns)


if __name__ == '__main__':
    main()
