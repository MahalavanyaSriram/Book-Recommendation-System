from collections import Counter, defaultdict
import math
import numpy as np
import os
import re
import pyspark.sql.functions as f
from scipy.sparse import csr_matrix
from pyspark.sql import SparkSession
import sys

def make_predictions(books, ratings_train, ratings_test):
    result = []
    for index,row in ratings_test.iterrows():
        mlist = list(ratings_train.loc[ratings_train['user_id'] == row['user_id']]['book_id'])
        csrlist = list(books.loc[books['book_id'].isin(mlist)]['features'])
        mrlist = list(ratings_train.loc[ratings_train['user_id'] ==row['user_id']]['rating'])
        cmlist = [cosine_sim(c,books.loc[books['book_id'] ==row['book_id']]['features'].values[0]) for c in csrlist]
        wan = sum([ v*mrlist[i] for i,v in enumerate(cmlist) if v>0 ])
        wadlist = [i for i in cmlist if i>0]
        if (len(wadlist)>0):
            result.append(wan/sum(wadlist))
        else:
            result.append(np.mean(mrlist))
    return np.array(result)


def cosine_sim(a, b):
    v1 = a.toarray()[0]
    v2  = b.toarray()[0]
    return sum(i[0] * i[1] for i in zip(v1, v2))/(math.sqrt(sum([i*i for i in v1]))*math.sqrt(sum([i*i for i in v2])))

def train_test_split(ratings):
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def featurize(books):   
    def tf(word,doc):
        return doc.count(word) / Counter(doc).most_common()[0][1]

    def df(word, doclist):
        return sum(1 for d in doclist if word in d)

    def tfidf(word, doc, dfdict, N):
        return tf(word, doc) * math.log10((N/dfdict[word]))

    def getcsrmatrix(tokens,dfdict,N,vocab):
        matrixRow_list = []
        matrixRow_list = np.zeros((1,len(vocab)),dtype='float')
        print("Initial matrix")
        print(matrixRow_list)
        for t in tokens:
            if t in vocab:
                matrixRow_list[0][vocab[t]] = tfidf(t,tokens,dfdict,N)
        print("Inside getcsrmatrix")
        print(matrixRow_list)
        return csr_matrix(matrixRow_list)

    N=books.count()
    doclist = list(books.select('tokens').toPandas()['tokens'])
    print("doclist generated")
    vocab = { i:x for x,i in enumerate(sorted(list(set(i for s in doclist for i in s)))) }
    print("vocab generated")

    dfdict = {}
    for v in vocab.items():
        dfdict[v[0]] = df(v[0],doclist)
    print("dfdict generated")

    print("calling getcsrmartix")
    books = books.rdd.map(lambda book:([book.book_id,book.title,book.tag_name,book.tokens,getcsrmatrix(book.tokens,dfdict,N,vocab)])).toDF(['book_id','title','tag_name','tokens','features'])

    return (books,vocab)

def mean_absolute_error(predictions, ratings_test):
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


if __name__ == "__main__":

    spark = (SparkSession
            .builder
            .appName("ContentBasedSystem")
            .getOrCreate())
    sc = spark.sparkContext
    sc.setLogLevel('WARN')

    print("===========Data Preprocessing started==========")

    # Loading data to Spark Data Frames
    books = spark.read.csv('data/books.csv', header = True)
    book_tags = spark.read.csv('data/book_tags.csv', header = True)
    tags = spark.read.csv('data/tags.csv', header = True)
    ratings = spark.read.csv('data/ratings_edit.csv', header = True)
    print("===========Data Loading complemeted==========")

    # Preprocessing the data
    tags_join = book_tags.join(tags, book_tags.tag_id == tags.tag_id)
    books_with_tags = books.join(tags_join, books.book_id == tags_join.goodreads_book_id)
    books_tags_unique = books_with_tags.groupBy("book_id").agg(f.concat_ws(" ",f.collect_list(books_with_tags.tag_name)))
    books_tags_unique = books_tags_unique.withColumnRenamed("concat_ws( , collect_list(tag_name))", "tag_name")
    books = books.join(books_tags_unique, books.book_id == books_tags_unique.book_id).drop(books_tags_unique.book_id)
    books = books.select("book_id", "title", "tag_name")
    books = books.dropDuplicates(["book_id"])
    print("===========Data Preprocessing complemeted==========")

    # Tokenizing the books
    books = books.rdd.map(lambda book:([book.book_id,book.title,book.tag_name,book.tag_name.split(' ')])).toDF(['book_id','title','tag_name','tokens'])
    print("===========Tokenizing complemeted==========")

    books, vocab = featurize(books)
    print("===========Featurize complemeted==========")
    ratings_train, ratings_test = ratings.randomSplit([.8,.2], seed=42)
    print("===========Splitting data to train & test complemeted==========")
    books.show(10)
    ratings_train, ratings_test = train_test_split(ratings)
    predictions = ratings_test.rdd.map(lambda row : make_predictions(books, ratings_train, row))
    predictions = make_predictions(books, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    spark.stop()