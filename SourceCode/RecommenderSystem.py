from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.sql.functions import rand

def load_movie_names():
    movie_names = {}
    skip_first = True
    with open("ml-latest/movies.csv") as f:
        for line in f:
            if skip_first:
                skip_first = False
                continue
            fields = line.split(",")
            movie_names[int(fields[0])] = fields[1]
    return movie_names

def get_movie_name(mov_rat):
    movie, rating = mov_rat
    return (movie_name_dictionary[movie], rating)

#generate rankings of movie ratings for RankingMetrics
def map1(ratings):
    actual = []
    preds = []
    for i in range(int(len(ratings)/2)):
        actual.append(ratings[2*i])
        preds.append(ratings[(2*i)+1])
    
    actual_indices = list(range(len(actual)))
    actual_indices.sort(key=lambda x: actual[x], reverse = True)
    actual_output = [0] * len(actual_indices)
    for i, x in enumerate(actual_indices):
        actual_output[x] = i + 1
    
    preds_indices = list(range(len(preds)))
    preds_indices.sort(key=lambda x: preds[x], reverse = True)
    preds_output = [0] * len(preds_indices)
    for i, x in enumerate(preds_indices):
        preds_output[x] = i + 1
    
    return (preds_output[:10], actual_output[:10]) #compare only the first 10 predictions and ground truths

conf = SparkConf().setMaster("local[*]").setAppName("Recommender System")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

text_file = sc.textFile("input/ml-latest/ratings.csv")

out1 = text_file.map(lambda line: line.strip().split(",")).zipWithIndex().filter(lambda tup: tup[1] > 1).map(lambda x: x[0])
#randomly shuffle the dataset
out2 = out1.toDF().orderBy(rand()).rdd.map(lambda tokens: Rating(int(tokens[0]),int(tokens[1]),float(tokens[2])))

ZippedData = out2.zipWithIndex()

#parameters for recommender system
rank = 50
num_iters = 10
lamda = 0.1
seed = 1
folds = 5
TSize = float(out2.count()/folds)
total_rmse = 0
total_mse = 0
total_map = 0

#5-fold cross-validation to get the best parameters for the recommender system
for i in range(folds):
    #splitting the train and test dataset
    CVTrainData = ZippedData.filter(lambda tup: tup[1]<int(TSize*i) or tup[1]>int(TSize*(i+1))).map(lambda x:x[0])
    CVTestData = ZippedData.filter(lambda tup: tup[1]>int(TSize*i) and tup[1]<int(TSize*(i+1))).map(lambda x:x[0])

    #training the model
    model = ALS.train(ratings = CVTrainData, rank = rank, iterations = num_iters, lambda_=lamda, seed = seed)

    #predicting on test dataset
    preds = model.predictAll(CVTestData.map(lambda p: (p[0], p[1]))).map(lambda r: ((r[0], r[1]), r[2]))

    ratesAndPreds = CVTestData.map(lambda r: ((r[0], r[1]), r[2])).join(preds)

    #evaluating predictions with actual ratings/rankings
    metrics = RegressionMetrics(ratesAndPreds.map(lambda r: r[1]))
    
    ratingPairs = ratesAndPreds.map(lambda r: (r[0][0], (r[1][0], r[1][1]))).reduceByKey(lambda x, y: x+y).map(lambda x : list(x[1]))
    rankAndPreds = ratingPairs.map(map1)
    rmetrics = RankingMetrics(sc.parallelize(rankAndPreds.collect())) #rankAndPreds is a PipelinedRDD and has to be converted into RDD

    MSE = metrics.meanSquaredError
    total_mse += MSE
    
    RMSE = metrics.rootMeanSquaredError
    total_rmse += RMSE
    
    MAP = rmetrics.meanAveragePrecision
    total_map += MAP

print("Average Mean Squared Error = " + str(total_mse/folds))
print("Average Root Mean Squared Error = " + str(total_rmse/folds))
print("Average Mean Average Precision = " + str(total_map/folds))

#This code is executed only after getting best parameters from Cross Validation

new_user_id = 0

#each tuple is (userID, movieID, rating)
new_user_ratings = [(0, 50,5),(0,100,4),(0,200,3),(0,300,3),(0,400,4),(0,500,4),(0,600,1),(0,700,1),
        (0,800,3),(0,900,5),(0,1000,4),(0,150,4),(0,250,3),(0,350,3),(0,450,4),(0,550,4),(0,650,1),
        (0,750,1),(0,850,3),(0,950,5),(0,458,1),(0,324,2),(0,114,3),(0,977,4),(0,260,5)]
out3 = sc.parallelize(new_user_ratings).map(lambda tokens: Rating(int(tokens[0]),int(tokens[1]),float(tokens[2])))

#appending new_user_ratings to the dataset
out4 = out2.union(out3)
new_model = ALS.train(ratings = out4, rank = rank, iterations = num_iters, lambda_=lamda, seed = seed)
new_user_movie_ids = map(lambda x: x[1], new_user_ratings)

#predicting ratings of movies not yet watched by the user
new_user_test = out2.filter(lambda x: x[1] not in new_user_movie_ids).map(lambda x: (new_user_id, x[1])).distinct()
new_user_recommendations = new_model.predictAll(new_user_test)

#recommending any 5 movies with a high rating (let's say rating more than 4)
movie_name_dictionary = load_movie_names()
new_user_recommended_ratings = new_user_recommendations.map(lambda x: (x[1], x[2])).map(get_movie_name).filter(lambda x: x[1] > 4.0)
print("Recommending 5 movies:")
print(new_user_recommended_ratings.take(5))
