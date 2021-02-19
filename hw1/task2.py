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

	rt = []
	for tweet in tweets:
		rt.append(tweet['retweet_count'])

	rt_rdd = sc.parallelize(rt)

	mean_rt = rt_rdd.mean()
	max_rt = rt_rdd.max()
	stdev_rt = rt_rdd.stdev()

	res = {}
	res['mean_retweet'] = mean_rt
	res['max_retweet'] = max_rt
	res['stdev_retweet'] = stdev_rt

	writeOutput(output_file, res)
	sc.stop()