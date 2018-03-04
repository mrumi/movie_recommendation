import sys, csv, math
from StringIO import StringIO
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, Rating
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics

APP_NAME = "My Spark Application"

def parse(row):	
	if row[0] == "userId":
		return
	else:	
		column = (int(row[0]),int(row[1]),float(row[2]))
		return column	
		
def split(line):
	reader = csv.reader(StringIO(line))
	return reader.next()

def main(sc):				
	ratings_info = sc.textFile("input/ratings.csv")
	ratings_data = ratings_info.map(split).map(parse).filter(lambda line: line!=None )	
		
	user_rating = sc.textFile("input/my_rating.csv").map(split).map(parse).filter(lambda line: line!=None )	
	
	userID = 0	
	rank = 12
	numIterations = 25
	
	train_data = ratings_data.union(user_rating)
	model = ALS.train(train_data, rank, iterations = numIterations, lambda_= 0.1)	
	
	ratedIds = user_rating.map(lambda x: x[1]).collect() # get movie IDs rated by new user
	#get movie IDs that are not rated
	unratedIds = ratings_data.filter(lambda x: x[0] not in ratedIds).map(lambda x: (userID, x[1]))	
	reco = model.predictAll(unratedIds)	
	#get movie ID and predicted rating
	predicted = reco.map(lambda x: (x.product, x.rating)).distinct()	
	#output = predicted.sortBy(keyfunc = lambda x: x[1], ascending=False)
	output = sc.parallelize(predicted.takeOrdered(5, key = lambda x: -x[1]))	
	output.saveAsTextFile("pred")				
	
if __name__ == "__main__":
	# Configure Spark
	conf = SparkConf().setAppName(APP_NAME)
	conf = conf.setMaster("local[*]")
	sc   = SparkContext(conf=conf)

	# Execute Main functionality
	main(sc)
