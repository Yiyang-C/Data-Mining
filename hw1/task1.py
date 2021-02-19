import sys
import json
from pyspark import SparkContext

def readInput(fp):
	data = []
	for line in open(fp, 'r'):
		data.append(json.loads(line))
	return data

def writeOutput(fp, data):
	with open(fp, 'w') as outfile:
		json.dump(data, outfile)


if __name__ == '__main__':
	input_file = sys.argv[1]
	output_file = sys.argv[2]
	# input_file = './data/Gamergate.json'
	# output_file = './data/testout.json'

	sc = SparkContext("local[*]", "PySpark Tutorial")

	tweets = readInput(input_file)
	n_tweet = len(tweets) # A.

	user_id = []
	for tweet in tweets:
		user_id.append(tweet['user']['id'])

	user_id_rdd = sc.parallelize(user_id)
	user_id_tuple_rdd = user_id_rdd.map(lambda x: (x, 1)).reduceByKey(lambda a, b: a + b)
	n_user = user_id_tuple_rdd.count() # B.

	user_name_fo = []
	for tweet in tweets:
		user_name_fo.append((tweet['user']['screen_name'], tweet['user']['followers_count']))
	
	user_name_fo_rdd = sc.parallelize(user_name_fo).sortBy(lambda x: x[1], False)
	popular_users_raw = user_name_fo_rdd.take(3)
	popular_users = [] # C.
	for user in popular_users_raw:
		popular_users.append([user[0], user[1]])

	weekday = []
	for tweet in tweets:
		weekday.append(tweet['created_at'].split()[0])

	weekday_rdd = sc.parallelize(weekday)
	weekday_tuple_rdd = weekday_rdd.map(lambda x: (x, 1)).reduceByKey(lambda a, b: a + b)
	weekday_cnt = weekday_tuple_rdd.collect()

	weekday_dict = {}
	for cnt in weekday_cnt:
		weekday_dict[cnt[0]]  = cnt[1]
	Tuesday_Tweet = weekday_dict['Tue'] # D.

	res = {}
	res['n_tweet'] = n_tweet
	res['n_user'] = n_user
	res['popular_users'] = popular_users
	res['Tuesday_Tweet'] = Tuesday_Tweet

	writeOutput(output_file, res)
	sc.stop()