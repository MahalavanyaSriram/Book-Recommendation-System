
import re
import sys

from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import ALS,MatrixFactorizationModel, Rating
import hashlib
from math import sqrt

# Get the id of book
def getId(x):
	y = re.match(r'(.*(.*?),){2}',x).group(1)[:-1]
	#yArr = y.split("\,\"\"")
	#y = yArr[0]
	#return y,x
	return y

# Returns a tuple with userId and another tuple of bookId and its corresponding rating that the user provided
def Generate_User_Book_Rating(dataLine):
	bookId = getId(dataLine)
	userId = re.match(r'((.*?),){1}',dataLine).group(2)[2:]
	#ratingValue = re.match(r'.*;"(.*?)"$',dataLine)
	ratingValue = dataLine.split(",")[2][-1]
	try:
		ratingScore = float(ratingValue)
		if(ratingScore == 0.0):
			ratingScore = 1.0
	except ValueError:
		ratingScore = 3.0
	return userId.encode('utf-8'),bookId.encode('utf-8'),str(ratingScore).encode('utf-8')

if __name__ == "__main__":
	if len(sys.argv) != 4:
		print >> sys.stderr, "Usage: program_file <Books_file> <User-Rating_file> <output_path> "
		exit(-1)
	conf = (SparkConf()
			.setMaster("local")
			.setAppName("als")
			.set("spark.executor.memory", "32g")
			.set("spark.driver.memory","32g"))
	sc = SparkContext(conf = conf)
	print("=================================== Read books ==============================================")
	# Read the books csv file and get those ids
	booksRDD = sc.textFile(sys.argv[1], 1)
	booksRDDheader = booksRDD.first()
	booksRDD = booksRDD.filter(lambda row: row != booksRDDheader)
	print("=================================== books read successfully  ==============================================")
	booksRDD = booksRDD.map(lambda x : getId(x))
	print("=================================== got book id successfully  ==============================================")
	print(booksRDD.collect())
	print("=================================== book id printed successfully  ==============================================")
	# Read the bookRatings csv file and get UserId, Bookid, Rating values
	userBookRatings = sc.textFile(sys.argv[2], 1)
	userbooksRDDheader = userBookRatings.first()
	userBookRatings = userBookRatings.filter(lambda row: row != userbooksRDDheader)
	print("===============================================================================")
	print(userBookRatings.first())
	print("===============================================================================")
	print("=================================== got user book ratings from ratings.csv  ==============================================")
	User_Book_Ratings = userBookRatings.map(lambda x : Generate_User_Book_Rating(x))
	print("=================================== generated user book retings  ==============================================")



	#print(User_Book_Rating.first())
	#creating a ratings object
	print("=================================== creating ratings object ===========================================================================")
	ratingsobj = User_Book_Ratings.map(lambda x: Rating(int(hashlib.sha1(x[0]).hexdigest(), 16)%(10 ** 6),int(hashlib.sha1(x[1]).hexdigest(), 16)%(10 ** 8), float(x[2])))
	#print "Ratings value: ##########################"
	#print(ratingsobj.first())

	#Generate a training and test set
	print("========================================= generating testing and train set ===========================================================")
	train_data, test_data = ratingsobj.randomSplit([0.8,0.2],None)

	#cache the data to speed up process

	train_data.cache()
	test_data.cache()

	#Setting up the parameters for ALS

	alsrank = 5 
	alsnumIterations = 10 

	#Build model on the training data
	print("============================================================ model training starts ====================================================")
	model = ALS.train(train_data, alsrank, alsnumIterations)

	# Reformat the train data
	print("=============================================predict==========================================================================================================")
	prediction_input = train_data.map(lambda x:(x[0],x[1]))

	#Returns Ratings(user, item, prediction)

	prediction = model.predictAll(prediction_input)

	# Reformat the test data
	test_input = test_data.map(lambda x:(x[0],x[1]))
	pred_test = model.predictAll(test_input)


	#Performance Evaluation

	#Organize the data to make (user, product) the key) for train data

	trained_values = train_data.map(lambda x:((x[0],x[1]), x[2]))
	predicted_values = prediction.map(lambda x:((x[0],x[1]), x[2]))

	#Join the trained_values and predicted_values

	train_output_values = trained_values.join(predicted_values)

	#Calculate Mean-Squared Error

	train_MSE = train_output_values.map(lambda r: (r[1][0] - r[1][1])**2).mean()
	print("Error for Training Set:")
	print(train_MSE)
	#Calculate Root-Mean-Squared Error
	train_RMSE = sqrt(train_MSE)
	print(train_RMSE)

	#Organize the data to make (user, product) the key) for train data

	test_values = test_data.map(lambda x:((x[0],x[1]), x[2]))
	test_predicted_values = pred_test.map(lambda x:((x[0],x[1]), x[2]))

	#Join the test_values and test_predicted_values
	test_output_values = test_values.join(test_predicted_values)

	#Calculate Mean-Squared Error
	test_MSE = test_output_values.map(lambda x: (x[1][0] - x[1][1])**2).mean()
	#Calculate Root-Mean-Squared Error
	test_RMSE = sqrt(test_MSE)
	print("Error for testing Set:")
	print(test_MSE)
	print(test_RMSE)

	#Save the Error value to a text file
	error = sc.parallelize([("Train MSE:",train_MSE),("Train RMSE:",train_RMSE),("Test MSE:",test_MSE),("Test RMSE:",test_RMSE)])
	error.coalesce(1).saveAsTextFile(sys.argv[3]+"/error.out")

	#Generate the recommendations for each user

	result = model.recommendProductsForUsers(10).collect()

	#result = model.recommendUsersForProducts(10)
	#result = model.recommendProducts(4361,20)
	print(result)
	#Save the result to a text file
	resultRDD = sc.parallelize(result)
	resultRDD.coalesce(1).saveAsTextFile(sys.argv[3]+"/result.out")


	#Convert Book Ids and User IDs to Integer to use

	user_ids = User_Book_Ratings.map(lambda k:(k[0],int(hashlib.sha1(k[0]).hexdigest(), 16)%(10 ** 6)))
	user_ids_list = user_ids.collect()

	# Save User ids to a file
	user_ids_RDD = sc.parallelize(user_ids_list)
	user_ids_RDD.coalesce(1).saveAsTextFile(sys.argv[3]+"/users.out")

	book_ids = User_Book_Ratings.map(lambda k:(k[1],int(hashlib.sha1(k[1]).hexdigest(), 16)%(10 ** 8)))
	book_ids_list = book_ids.collect()

	# Save Book ids to a file
	book_ids_RDD = sc.parallelize(book_ids_list)
	book_ids_RDD.coalesce(1).saveAsTextFile(sys.argv[3]+"/books.out")

	sc.stop()
