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
	
	fold1, fold2, fold3, fold4, fold5 = ratings_data.randomSplit([0.2, 0.2, 0.2, 0.2, 0.2])
	folds = [fold1, fold2, fold3, fold4, fold5]	
	
	rank = 12
	itr = 25
	mse = 0
	rmse = 0
	map = 0
	for i in range(5):
		test_data = folds[i]
		train_data = sc.emptyRDD()
		for j in range(5):
			if i == j:
				continue
			else:
				train_data = train_data.union(folds[j])
		
		model = ALS.train(train_data, rank, iterations = itr, lambda_= 0.1)			
		testdata = test_data.map(lambda p: (p[0], p[1]))	
		predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))		
		rates = test_data.map(lambda r: ((r[0], r[1]), r[2]))	
		predsAndlabels = predictions.join(rates).map(lambda tup: tup[1])	
		actual_rating= predsAndlabels.map(lambda r: r[1]).collect()
		predicted_rating = predsAndlabels.map(lambda r: r[0]).collect()
		predAndReal = sc.parallelize([(predicted_rating, actual_rating)])
	
		metrics = RegressionMetrics(predsAndlabels)
		metric = RankingMetrics(predAndReal)
		mse += metrics.meanSquaredError
		rmse += metrics.rootMeanSquaredError
		map += metric.meanAveragePrecision
		
	k_mse = mse/5.0
	k_rmse = rmse/5.0
	k_map = map/5.0
	print("MSE = %s" % k_mse)
	print("RMSE = %s" % k_rmse)
	print("MAP = %s" % k_map)
				
if __name__ == "__main__":
	# Configure Spark
	conf = SparkConf().setAppName(APP_NAME)
	conf = conf.setMaster("local[*]")
	sc   = SparkContext(conf=conf)

	# Execute Main functionality
	main(sc)
