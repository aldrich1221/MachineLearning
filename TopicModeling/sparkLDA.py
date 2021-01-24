

# gsutil cp desktop/sparkPLSA.py gs://electricvehiclestorage/pyspark_dataproc_example/plsa/sparkPLSA.py
# gcloud dataproc jobs submit pyspark --cluster=electricvehicle  gs://electricvehiclestorage/pyspark_dataproc_example/plsa/sparkPLSA.py --region=asia-east1

import sys
from pyspark.ml.feature import CountVectorizer,CountVectorizerModel,Tokenizer,StopWordsRemover,HashingTF,IDF
from pyspark.mllib.linalg import Vector, Vectors
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.ml.clustering import LDA

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
from google.cloud import storage


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
        filter_string_udf = udf(lambda x: x.replace("\\n","").replace("\n"," ").replace(","," "), StringType())
        df = df.withColumn('temp', filter_string_udf(col('temp')))

        df=df.withColumn('temp', regexp_replace('temp', specificRule, ''))
        
        df=df.withColumn(self.outputCol, regexp_replace('temp', stringRule, ''))
        
        return df
class CustomizedFilter(Transformer):
  
    def __init__(self, inputCol, outputCol):
        super(CustomizedFilter, self).__init__()
        self.inputCol = inputCol
        self.outputCol = outputCol
    def _transform(self, df):
    	
        filter_length_udf = udf(lambda row: [x for x in row if len(x) >= 3], ArrayType(StringType()))
        df = df.withColumn(self.outputCol, filter_length_udf(col(self.inputCol)))
       	
        
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


class Keywords:
	def __init__(self, classId, ori_keyword, extend_keyword):
		self.classId = classId
		self.ori_keyword = ori_keyword
		self.extend_keyword = extend_keyword
	def __repr__(self):
		return repr((self.classId, self.ori_keyword, self.extend_keyword))
	def getExtendKeyword(self):
        
		return self.extend_keyword
	def getClassId(self):
		return self.classId

def ExtendKeyword(sc,lda_model,vocabArray,groupframe):
	topics_keywords=CheckTopReleventWord(sc,lda_model,vocabArray)
	ReleventWord_list_index = groupframe
	ReleventWords=[]
	
	
	[ReleventWords.append(vocabArray[[i for i, e in enumerate(row[1].toArray()) if e != 0.0][0]]) for row in ReleventWord_list_index]
	
	FinalKeywordDict=dict()
	FinalKeywordList=list()
	for originGroupsIndex in range(len(ReleventWords)):
		relevant_word=ReleventWords[originGroupsIndex]
		findit=False
		for topicIndex in range(len(topics_keywords)):
			keywordListPerTopic=topics_keywords[topicIndex]
			for w in keywordListPerTopic:
				
				if relevant_word==w: #match!!
					print("match!! ",relevant_word ,w)
					FinalKeywordDict[originGroupsIndex]=keywordListPerTopic

					FinalKeywordList.append(Keywords(originGroupsIndex,relevant_word,keywordListPerTopic))
					findit=True
					break;
			if findit==True:
				break;
	FinalSortedKeywordList=sorted(FinalKeywordList, key = lambda s: s.classId)
	rdd = sc.parallelize(FinalSortedKeywordList)
	
	return FinalSortedKeywordList,rdd

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
				#print("match!! ",relevant_word ,array[0],i)
				maxValue=0
				chooseIndex=0
				for j in range(1,len(array)):
					if maxValue<float(array[j]):

						if (j not in TopicIndex):
							maxValue=float(array[j])
							chooseIndex=j
						
				break

		TopicIndex.append(chooseIndex)

	
	DocClassifyTopic=[]
	testindex=0
	

	for i_doc in range(len(doc_topic)):
		valueArray=[float(x) for x in doc_topic[i_doc].split(' ')]
		tempValue=0
		m = max(valueArray)
		index=[i for i, j in enumerate(valueArray) if j == m] #lagest topic

		for i_topicindex in range(len(TopicIndex)):
			#print(index," vs ",i_topicindex,TopicIndex[i_topicindex])
			if index[0]==TopicIndex[i_topicindex]:
				
				DocClassifyTopic.append(i_topicindex)
				break
		
	




def ComputeSimilarity(sc,sqlContext,documentRdd,KeywordsData,preProcessedModel):
	from pyspark.mllib.linalg.distributed import IndexedRowMatrix
	from pyspark.mllib.linalg import Vectors as MLlibVectors
	

	
	print("Compute Similarity....")
	
	KeywordsList=list()
	for row in KeywordsData:
		KeywordsList.append((row.getClassId(),row.getExtendKeyword()))
	schema = StructType([
	    StructField('ClassId', IntegerType(), True),
	    StructField('extend_keyword', ArrayType(StringType()), True)
	])

	
	RDD=sc.parallelize(KeywordsList)
	
	KeywordsDataFrame=sqlContext.createDataFrame(RDD,schema)

	
	KeywordsDataFrame.show(n=2)

	KeywordsDataFrame=KeywordsDataFrame.selectExpr("ClassId as doc_id","extend_keyword as fiteredContent")
	KeywordsDataFrame=KeywordsDataFrame.select("doc_id","fiteredContent")
	
	KeywordsDataFrame.show(n=2)
	

	KeyWordTFIDF=preProcessedModel.stages[7].transform(preProcessedModel.stages[6].transform(KeywordsDataFrame))

	KeyWordData=KeyWordTFIDF.select("doc_id","features").rdd.mapValues(MLlibVectors.fromML).map(list)
	
	documentList=documentRdd.map(list).collect()
	keyWordList=KeyWordData.map(list).collect()
	

	Final=list()
	for sparseVec1Index in range(len(documentList)):
		sparseVec1=documentList[sparseVec1Index][1]
		sparseVecNorm1=sparseVec1.norm(2)
		MAXsimilarity=-1
		topic=0
		for sparseVec2Index in range(len(keyWordList)):
			sparseVec2=keyWordList[sparseVec2Index][1]
			sparseVecNorm2=sparseVec2.norm(2)
			
				
			similarity=sparseVec1.dot(sparseVec2)/np.sqrt(sparseVecNorm1*sparseVecNorm2)
			if similarity>MAXsimilarity:
				MAXsimilarity=similarity
				topic=sparseVec2Index
		#Final.append((sparseVec1Index,topic))
		Final.append(topic)

	print("==========Final Output===========")
	print(Final)




		
def CheckTopReleventWord(sc,lda_model,vocabArray):
	wordNumbers = 200
	topicIndices = sc.parallelize(lda_model.describeTopics(maxTermsPerTopic = wordNumbers))
	def topic_render(topic):
	    terms = topic[0]
	    result = []
	    for i in range(wordNumbers):
	        term = vocabArray[terms[i]]
	        result.append(term)
	    return result
	topics_final = topicIndices.map(lambda topic: topic_render(topic)).collect()
	# for topic in range(len(topics_final)):
	#     #print ("Topic" + str(topic) + ":")
	#     for term in topics_final[topic]:
	#         print (term)
	#     print ('\n')
	return topics_final
	#print(lda_model.getTopicDistributionCol(5))

def main():
	iterations=1
	if len(sys.argv)>1:
		iterations=sys.argv[1]
	#sparkConfig=SparkConf().setAppName("PLSA").setMaster('yarn')
	#sc = SparkContext(pyFiles=["gs://electricvehiclestorage/pyspark_dataproc_example/plsa/KeywordsClass.py"])
	sc = SparkContext()
	sqlContext = SQLContext(sc)
	precessedData,groupframe,preProcessedModel=preProcessPipeline(sc,sqlContext)
	num_topics = 17
	max_iterations = 50
	#precessedData.printSchema()
	from pyspark.mllib.linalg import Vectors as MLlibVectors
	from pyspark.mllib.clustering import LDA as MLlibLDA

	documentData=precessedData.select("doc_id", "features").rdd.mapValues(MLlibVectors.fromML).map(list)
	documentDataList=documentData.collect()
	
	groupDataList=groupframe.select("class_id", "relevant_words").rdd.mapValues(MLlibVectors.fromML).map(list).collect()

	lda_model=MLlibLDA.train(documentData, k=num_topics, maxIterations=max_iterations)
	

	TopicTermMatrix=lda_model.topicsMatrix()
	
	
	vocabArray=preProcessedModel.stages[6].vocabulary
	FinalKeywordList,KeywordRdd=ExtendKeyword(sc,lda_model,vocabArray,groupDataList)


	ComputeSimilarity(sc,sqlContext,documentData,FinalKeywordList,preProcessedModel)



def preProcessPipeline(sc,sqlc):
	StopWordFilePath="gs://electricvehiclestorage/pyspark_dataproc_example/plsa/stop.txt"
	stopword = sqlc.read.format('com.databricks.spark.csv').options(header='false', inferschema='true').load(StopWordFilePath)#
	
	stopword=stopword.selectExpr("_c0 as stopWords")
	
	stopwordList=list(stopword.select('stopWords').toPandas()['stopWords']) # => [1, 2, 3, 4]
	stopwordList.append('')
	#print(stopwordList)
	def build_pipeline():
		
		textcleaner = TextClean(inputCol='content', outputCol='content_clean')
		tokenizer = Tokenizer(inputCol='content_clean', outputCol='content_token')
		
		
		
		defaultRemover= StopWordsRemover(inputCol='content_token', outputCol='ContentWithoutDesultStopwords')
		coutomizedRemover=StopWordsRemover(inputCol='ContentWithoutDesultStopwords', outputCol='ContentWithoutStopwords' ,stopWords=stopwordList)
		
		textstemmer = Stemmer(inputCol='ContentWithoutStopwords', outputCol='StemmedContent')
		
		customizedFilter=CustomizedFilter(inputCol='StemmedContent', outputCol="fiteredContent")
		TFcountvectorizer=CountVectorizer(inputCol="fiteredContent", outputCol="countvectorizedContent", vocabSize=5000, minDF=10.0)
		idf = IDF(inputCol="countvectorizedContent", outputCol="features")
		
		pipeline = Pipeline(stages=[textcleaner,tokenizer,defaultRemover,coutomizedRemover,textstemmer,customizedFilter,TFcountvectorizer,idf])
		return pipeline

	dataFilePath="gs://electricvehiclestorage/pyspark_dataproc_example/plsa/doc.csv"
	groupFilePath="gs://electricvehiclestorage/pyspark_dataproc_example/plsa/groups.csv"

	dataframe = sqlc.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(dataFilePath)
	groupframe = sqlc.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(groupFilePath)

	
	print("Documents Number:",dataframe.count())
	print("Groups Number:",groupframe.count())

	pipeline=build_pipeline()
	preProcessedModel=pipeline.fit(dataframe)
	
	
	
	precessedData=preProcessedModel.transform(dataframe).select('doc_id','features')
	
	print("precessedData Number:",precessedData.count())


	groupframe = groupframe.selectExpr("class_id as doc_id", "relevant_words as content")
	precessedGroupFrame=preProcessedModel.transform(groupframe).select('doc_id','features')
	precessedGroupFrame=precessedGroupFrame.selectExpr("doc_id as class_id", "features as relevant_words")


	return precessedData,precessedGroupFrame,preProcessedModel



if __name__=="__main__":
	main()