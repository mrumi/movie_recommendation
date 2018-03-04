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
	
	training, validation, test = ratings_data.randomSplit([6, 2, 2])
	validation_data = validation.map(lambda x: (x[0], x[1]))
	test_data = test.map(lambda x: (x[0], x[1]))
	
	ranks = [6, 8, 10, 12, 14]	
	iteration = 10 
	
	min_error = float('inf')	
	best_rank = -1
	string=""
	for rank in ranks:
		model = ALS.train(training, rank, iterations = iteration, lambda_= 0.1)
		predictions = model.predictAll(validation_data).map(lambda r: ((r[0], r[1]), r[2]))
		ratings = validation.map(lambda r: ((r[0], r[1]), r[2]))
		preds_and_rates = predictions.join(ratings)
		predsAndratess = preds_and_rates.map(lambda tup: tup[1])	
		metrics = RegressionMetrics(predsAndratess)
		error = metrics.rootMeanSquaredError
		
		string+= "For rank "+ str(rank) +"the RMSE is " +str(error)+"\n"
		if error < min_error:
			min_error = error
			best_rank = rank
	
	string+= "The best model was trained with rank " +str(best_rank) +"\n"	
		
	model = ALS.train(training, best_rank, iterations=iteration,lambda_= 0.1)
	predictions = model.predictAll(test_data).map(lambda r: ((r[0], r[1]), r[2]))
	ratings = test.map(lambda r: ((r[0], r[1]), r[2]))
	preds_and_rates = predictions.join(ratings)
	predsAndratess = preds_and_rates.map(lambda tup: tup[1])	
	metrics = RegressionMetrics(predsAndratess)
	error = metrics.rootMeanSquaredError
	
	string+= "The RMSE for Test data is " + str(error) + "\n"
	print string	
	
	
if __name__ == "__main__":
	# Configure Spark
	conf = SparkConf().setAppName(APP_NAME)
	conf = conf.setMaster("local[*]")
	sc   = SparkContext(conf=conf)

	# Execute Main functionality
	main(sc)
