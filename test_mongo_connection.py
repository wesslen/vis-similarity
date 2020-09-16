import pymongo
from pymongo import MongoClient
from pandas import json_normalize

# add in <user> and <pwd>
client = pymongo.MongoClient("mongodb://gt-user:georgiatech@cluster0-shard-00-00.c1tlg.mongodb.net:27017,cluster0-shard-00-01.c1tlg.mongodb.net:27017,cluster0-shard-00-02.c1tlg.mongodb.net:27017/<dbname>?ssl=true&replicaSet=atlas-6nhyvn-shard-0&authSource=admin&retryWrites=true&w=majority")

# Database Names
print("Database Names", client.database_names())

# Choose the vissimilarity database
db = client.vissimilarity

# Get the documents and print
docs = db['docs']
df = json_normalize(list(docs.find()))
print(df.head())
