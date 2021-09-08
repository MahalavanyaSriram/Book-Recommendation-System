
Execution Details for Book Recommendation Systems
=================================================

Project folder structure
========================
- Collaborative
   - StochasticGradientDescent.ipynb
   - ALS
      - ALS.py
      - output
		
- Content_based
   - Content based system.ipynb
   - spark
      - ContentBasedSystems.py

Running the .ipynb files
=========================
1. Have the necessary data files in the appropriate location
2. Run the the notebook as it is

Running the spark ALS files
===========================

We executed the ALS part in Amazon Web Services.

1. First, created a bucket and loaded the bucket with the input files and with the source code.
2. Create an cluster with required number of nodes and connect to it through the ssh command given in description(assuming we are using MAC)
3. Then we downloaded the inputs from the AWS into the hadoop by using the following commands:
	$ aws s3 cp s3://bucketname/books.csv ./
	$ aws s3 cp s3://bucketname/ratings.csv ./
	$ aws s3 cp s3://bucketname/ALS.py ./
4. $ spark-submit ALS.py books.csv ratings.csv output
5. To extract the files from the bucket , We downloaded from the bucket
	The output consists of error and result file. 
	Error file consists of root mean square error for the test and the train data.
	Result file consists of the top ten recommendations for each user.

Running the content based in spark
==================================

We tried to execute the code both in AWS and dsba cluster.
We are getting the following error in execution. ("Scipy module not found")
On detailed exploration of the error. It occurs as the module gets downloaded and accessible by the master node only and not by the worker node. We tried to rectify the issue by including custom bootstrap in aws cluster. But all our efforts were in vain. 

Report url
==========

https://webpages.uncc.edu/cmannem/cloud.html



