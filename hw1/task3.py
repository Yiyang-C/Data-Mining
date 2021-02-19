import sys
import json
from pyspark import SparkContext

def readInput(fp):
	data = open(fp, "r", encoding='UTF-8')
	return data

def writeOutput(fp, data):
	with open(fp, 'w') as outfile:
		json.dump(data, outfile)


if __name__ == '__main__':
	input_file = sys.argv[1]
	output_file = sys.argv[2]
	# input_file = './data/tweets'
	# output_file = './data/testout5.json'

	sc = SparkContext("local[*]", "PySpark Tutorial")

	# tweets = readInput(input_file)

	tweets_rdd = sc.textFile(input_file)
	chunk_count = tweets_rdd.count() # C.

	word_list_rdd = tweets_rdd.flatMap(lambda x: x.split(' '))

	word_list_rdd_map = word_list_rdd.map(lambda x: (x, 1))

	word_cnt_rdd = word_list_rdd_map.reduceByKey(lambda a, b: a + b)

	word_cnt_rdd_sort = word_cnt_rdd.sortBy(lambda x: x[1], False)
	tmp = word_cnt_rdd_sort.take(1)
	max_word = [tmp[0][0], tmp[0][1]] # A.

	word_mindless = word_list_rdd.filter(lambda x: x == 'mindless')
	mindless_count = word_mindless.count() # B.

	res = {}
	res['max_word'] = max_word
	res['mindless_count'] = mindless_count
	res['chunk_count'] = chunk_count

	writeOutput(output_file, res)
	sc.stop()