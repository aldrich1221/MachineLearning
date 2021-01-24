

# gsutil cp desktop/sparkPLSA.py gs://electricvehiclestorage/pyspark_dataproc_example/plsa/sparkPLSA.py
# gcloud dataproc jobs submit pyspark --cluster=electricvehicle  gs://electricvehiclestorage/pyspark_dataproc_example/plsa/sparkPLSA.py --region=asia-east1

import sys
from pyspark.ml.feature import CountVectorizer,CountVectorizerModel,Tokenizer,StopWordsRemover,HashingTF,IDF

from pyspark import RDD
import numpy as np
from numpy.random import RandomState
#from PLSA import PLSA
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import types
from pyspark.sql.types import *

from pyspark.sql.functions import udf, col, lower, regexp_replace
from pyspark.ml import Pipeline, Transformer
from pyspark import SparkConf, SparkContext
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark import keyword_only
from pyspark.sql import DataFrame
import re
import numpy as np
from numpy.random import RandomState
import pandas
from nltk.stem.snowball import SnowballStemmer
from subprocess import call
import pandas as pd


class PLSA:

	def __init__(self, data, sc, k, is_test=False, max_itr=1000, eta=1e-6):

		
		self.max_itr = max_itr
		self.k = sc.broadcast(k)
		self.originalData=data.rdd.map(list).map(lambda xx:xx[0])
		self.sc = sc
		self.eta = eta
		self.rd = sc.broadcast(RandomState(1) if is_test else RandomState())

	def train(self):

		
		self.word_dict_b = self.initializeDict()
		
		self.makeDoc2Word()
		
		self.initializeDistributionWordTopic()

		Previousloglikelyhood= self.loglikelyhood()

		print("Demension: ",self.k.value,self.v.value)

		print "L(%d)=%.5f" %(0,Previousloglikelyhood)

		for i in range(self.max_itr):
			print("iteration  ",i,len(self.data.flatMap(lambda d: d).collect()))
			self.EStep()
			
			self.MStep()
			Newloglikelyhood = self.loglikelyhood()

			improve = np.abs((Previousloglikelyhood-Newloglikelyhood)/Previousloglikelyhood)
			Previousloglikelyhood = Newloglikelyhood

			print "L(%d)=%.5f with %.6f%% improvement" %(i+1,Newloglikelyhood,improve*100)
			if improve <self.eta:
				break

	def MStep(self):
		
		k = self.k
		v = self.v

		def updateDistributionDocTopic(doc):
			
			doc['topic'] = doc['topic'] - doc['topic']

			topic_doc = doc['topic']
			words = doc['words']
			for (word_index,word) in words.items():
				topic_doc += word['count']*word['topic_word']
			topic_doc /= np.sum(topic_doc)
			print("MSteop len(topic_doc)",len(topic_doc))
			return {'words':words,'topic':topic_doc}

		self.data = self.data.map(updateDistributionDocTopic)
	
		self.data.cache()

		def updateDistributionWordTopic(doc):
			
			probility_word_given_topic = np.matrix(np.zeros((k.value,v.value)))

			words = doc['words']
			for (word_index,word) in words.items():
				probility_word_given_topic[:,word_index] += np.matrix(word['count']*word['topic_word']).T

			return probility_word_given_topic

		probility_word_given_topic = self.data.map(updateDistributionWordTopic).sum()
		probility_word_given_topic_row_sum = np.matrix(np.sum(probility_word_given_topic,axis=1))

		
		probility_word_given_topic = np.divide(probility_word_given_topic,probility_word_given_topic_row_sum)

		self.probility_word_given_topic = self.sc.broadcast(probility_word_given_topic)

	def EStep(self):
		
		probility_word_given_topic = self.probility_word_given_topic
		k = self.k

		def updateDistributionWordTopicWord(doc):
			topic_doc = doc['topic']
			words = doc['words']

			for (word_index,word) in words.items():
				topic_word = word['topic_word']
				for i in range(k.value):
					topic_word[i] = probility_word_given_topic.value[i,word_index]*topic_doc[i]
					
				topic_word /= np.sum(topic_word)
			return {'words':words,'topic':topic_doc}

		self.data = self.data.map(updateDistributionWordTopicWord)

	def  initializeDistributionWordTopic(self):
	
		m = self.v.value

		probility_word_given_topic = self.rd.value.uniform(0,1,(self.k.value,m))
		probility_word_given_topic_row_sum = np.matrix(np.sum(probility_word_given_topic,axis=1)).T

		
		probility_word_given_topic = np.divide(probility_word_given_topic,probility_word_given_topic_row_sum)

		self.probility_word_given_topic = self.sc.broadcast(probility_word_given_topic)

	def makeDoc2Word(self):

		word_dict_b = self.word_dict_b
		k = self.k
		rd = self.rd
	
		def wordCountDoc(doc):
			wordcount ={}
			word_dict = word_dict_b.value
			for word in doc:
				if wordcount.has_key(word_dict[word]):
					wordcount[word_dict[word]]['count'] += 1
				else:
				
					wordcount[word_dict[word]] = {'count':1,'topic_word': rd.value.uniform(0,1,k.value)}

			topics = rd.value.uniform(0, 1, k.value)
			topics = topics/np.sum(topics)
			return {'words':wordcount,'topic':topics}

		self.data = self.originalData.map(wordCountDoc)

	def initializeDict(self):
	
		
		words = self.originalData.flatMap(lambda d: d).distinct().collect()
		allwords=self.originalData.flatMap(lambda d: d).collect()
		
		word_dict = {w: i for w, i in zip(words, range(len(words)))}
		self.v = self.sc.broadcast(len(word_dict))
		return self.sc.broadcast(word_dict)

	def loglikelyhood(self):
		probility_word_given_topic = self.probility_word_given_topic
		k = self.k

		def likelyhood(doc):
			l = 0.0
			topic_doc = doc['topic']
			words = doc['words']

			for (word_index,word) in words.items():
				l += word['count']*np.log(np.matrix(topic_doc)*probility_word_given_topic.value[:,word_index])
			return l
		return self.data.map(likelyhood).sum()

	def save(self,f_word_given_topic,f_doc_topic):
	


		doc_topic = self.data.map(lambda x:' '.join([str(q) for q in x['topic'].tolist()])).collect()
		probility_word_given_topic = self.probility_word_given_topic.value

		word_dict = self.word_dict_b.value
		word_given_topic = []

		for w,i in word_dict.items():
			word_given_topic.append('%s %s' %(w,' '.join([str(q[0]) for q in probility_word_given_topic[:,i].tolist()])))



		return word_given_topic,doc_topic

class TextClean(Transformer):
  
    def __init__(self, inputCol, outputCol):
        super(TextClean, self).__init__()
        self.inputCol = inputCol
        self.outputCol = outputCol
    def _transform(self, df):
    	
        emailRule = '\S+@\S+'
        stringRule="[^a-zA-Z\\s]"
        specificRule='Subject:'

        df=df.withColumn('temp', regexp_replace(self.inputCol, emailRule, ''))
        filter_length_udf = udf(lambda x: x.replace("\\n","").replace("\n"," ").replace(","," "), StringType())
        df = df.withColumn('temp', filter_length_udf(col('temp')))
        df=df.withColumn('temp', regexp_replace('temp', specificRule, ''))
        df=df.withColumn(self.outputCol, regexp_replace('temp', stringRule, ''))
        
        return df

class Stemmer(Transformer):

    def __init__(self, inputCol, outputCol):
        super(Stemmer, self).__init__()
        self.inputCol = inputCol
        self.outputCol = outputCol
	

    def _transform(self, df):
    	stemmer = SnowballStemmer(language='english')
    	stemmer_udf = udf(lambda tokens: [stemmer.stem(token) for token in tokens], ArrayType(StringType()))
    	df = df.withColumn(self.outputCol, stemmer_udf(self.inputCol)).select('doc_id', self.outputCol)
    	return df

def refinedSeedWords(word_given_topic,doc_topic,groupframe):
	

	ReleventWord_list = groupframe.select('StemmedContent').collect()
	ReleventWords=[]
	[ReleventWords.append(row.StemmedContent[0]) for row in ReleventWord_list]
	
	TopicIndex=[]
	maxValue=0
	chooseIndex=0
	for relevant_word in ReleventWords:
		for i in range(len(word_given_topic)):
			array=word_given_topic[i].split(' ')

			if relevant_word==array[0]: #match!!
				print("match!! ",relevant_word ,array[0],i)
				maxValue=0
				chooseIndex=0
				for j in range(1,len(array)):
					if maxValue<float(array[j]):

						if (j not in TopicIndex):
							maxValue=float(array[j])
							chooseIndex=j
						
				break

		TopicIndex.append(chooseIndex)

	print(TopicIndex)
	DocClassifyTopic=[]
	testindex=0

	for i_doc in range(len(doc_topic)):
		valueArray=[float(x) for x in doc_topic[i_doc].split(' ')]
		tempValue=0
		m = max(valueArray)
		index=[i for i, j in enumerate(valueArray) if j == m] 

		for i_topicindex in range(len(TopicIndex)):
			
			if index[0]==TopicIndex[i_topicindex]:
				
				DocClassifyTopic.append(i_topicindex)
				break
		
			
	print(DocClassifyTopic)





	


def main():
	iterations=1
	if len(sys.argv)>1:
		iterations=sys.argv[1]
	
	sc = SparkContext()
	sqlContext = SQLContext(sc)
	precessedData,groupframe=preProcessPipeline(sc,sqlContext)
	plsa = PLSA(precessedData,sc,groupframe.rdd.count(),max_itr=int(iterations))
	plsa.train()
	word_given_topic,doc_topic=plsa.save("gs://electricvehiclestorage/pyspark_dataproc_example/output/topic_word.txt","gs://electricvehiclestorage/pyspark_dataproc_example/output/doc_topic.txt")
	print(np.array(doc_topic).shape,np.array(word_given_topic).shape)

	refinedSeedWords(word_given_topic,doc_topic,groupframe)

def preProcessPipeline(sc,sqlc):
	stopwordList=['']

	def build_pipeline():
		
		textcleaner = TextClean(inputCol='content', outputCol='content_clean')
		tokenizer = Tokenizer(inputCol='content_clean', outputCol='content_token')
		
		
		
		defaultRemover= StopWordsRemover(inputCol='content_token', outputCol='ContentWithoutDesultStopwords')
		coutomizedRemover=StopWordsRemover(inputCol='ContentWithoutDesultStopwords', outputCol='ContentWithoutStopwords' ,stopWords=stopwordList)
		
		textstemmer = Stemmer(inputCol='ContentWithoutStopwords', outputCol='StemmedContent')

		pipeline = Pipeline(stages=[textcleaner,tokenizer,defaultRemover,coutomizedRemover,textstemmer])
		return pipeline

	dataFilePath="gs://electricvehiclestorage/pyspark_dataproc_example/plsa/doc.csv"
	groupFilePath="gs://electricvehiclestorage/pyspark_dataproc_example/plsa/groups.csv"
	
	dataframe = sqlc.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(dataFilePath)
	groupframe = sqlc.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(groupFilePath)

	

	pipeline=build_pipeline()
	preProcessedModel=pipeline.fit(dataframe)
	precessedData=preProcessedModel.transform(dataframe).select('StemmedContent')

	groupframe = groupframe.selectExpr("class_id as doc_id", "relevant_words as content")
	precessedGroupFrame=preProcessedModel.transform(groupframe).select('StemmedContent')
	


	return precessedData,precessedGroupFrame


if __name__=="__main__":
	main()