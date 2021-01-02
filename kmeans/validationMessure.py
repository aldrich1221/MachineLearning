def readFile(fileName):
	f = open(fileName, mode='r')
	datatxt=[]
	for line in f.readlines():
		number=line.replace('\n','').split(' ')

		datatxt.append(float(number[1]))
	
	f.close
	return datatxt
def writeFile(filename,textdata):
	StringList=[]
	count=0
	for ans in textdata:
		StringList.append(str(ans[0])+" "+str(ans[1])+"\n")
		count=count+1
	f=open(filename,mode='w')
	
	f.writelines(StringList)
	f.close()


def SkLearn(clustering_1,clustering_2,clustering_3,clustering_4,clustering_5,partitions):
	from sklearn.metrics.cluster import normalized_mutual_info_score
	import sklearn.metrics
	import numpy as np
	from sklearn.metrics import jaccard_similarity_score
	import numpy as np
	Ans=np.zeros((5,2))
	Ans[0,0]=normalized_mutual_info_score(clustering_1, partitions)
	Ans[0,1]=Jaccard_coef(clustering_1,partitions)

	Ans[1,0]=normalized_mutual_info_score(clustering_2, partitions)
	Ans[1,1]=Jaccard_coef(clustering_2,partitions)

	Ans[2,0]=normalized_mutual_info_score(clustering_3, partitions)
	Ans[2,1]=Jaccard_coef(clustering_3,partitions)

	Ans[3,0]=normalized_mutual_info_score(clustering_4, partitions)
	Ans[3,1]=Jaccard_coef(clustering_4,partitions)

	Ans[4,0]=normalized_mutual_info_score(clustering_5, partitions)
	Ans[4,1]=Jaccard_coef(clustering_5,partitions)
	return Ans.tolist()


	
def calculatePossibilitiesForTruthAssignment(TrueY,PredictY):
	from scipy.special import comb

	def confusionDict(TrueY,PredictY):
		from collections import Counter
		import numpy as np
		TrueYDict=Counter(TrueY)
		PredictYDict=Counter(PredictY)
		confusionDict=Counter(zip(TrueY,PredictY))
		return TrueYDict,PredictYDict,confusionDict

	trueYDict,predictYDict,confusionDict=confusionDict(TrueY,PredictY)
	
	#calculate True Positive
	total=0
	truePositive=0
	for key,value in confusionDict.items():
		truePositive+=value**2
		total+=value
	truePositive=0.5*(truePositive-total)

	#calculate False Negative
	falseNegative=0
	for key,value in trueYDict.items():
		falseNegative += comb(value,2)
	falseNegative -= truePositive

	#calculate False Positive
	falsePositive=0
	for key,value in predictYDict.items():
		falsePositive += comb(value,2)
	falsePositive -= truePositive

	#calculate True Negative
	trueNegative=total-truePositive-falseNegative-falsePositive

	return truePositive,falseNegative,falsePositive,trueNegative
def Jaccard(TrueY,PredictY):
	TP,FN,FP,TN=calculatePossibilitiesForTruthAssignment(TrueY,PredictY)
	Jaccard = TP/(TP+FN+FP)
	return Jaccard

def normalizedMutualInformation(TrueY,PredictY):
	def Entropy(labelsList):
		from collections import Counter
		import numpy as np
		total=len(labelsList)
		entropy=0
		labelDict=Counter(labelsList)
		for value in labelDict.values():
			p=float(value)/float(total)
			entropy-=p*np.log(p)
		return entropy

	def mutualInformation(TrueY,PredictY):

		def confusionDict(TrueY,PredictY):
			from collections import Counter
			import numpy as np
			TrueYDict=Counter(TrueY)
			PredictYDict=Counter(PredictY)
			confusionDict=Counter(zip(TrueY,PredictY))
			return TrueYDict,PredictYDict,confusionDict
		
		trueYDict,predictYDict,confusionDict=confusionDict(TrueY,PredictY)
		mutualInformation=0
		total=len(TrueY)
		
		for key,value in confusionDict.items():

			t,c = key
			
			pij = float(value)/float(total)
			pc = float(predictYDict[c])/float(total)
			pt = float(trueYDict[t])/float(total)
			mutualInformation += pij*np.log(pij/(pc*pt))
			

		return mutualInformation

	
	I = mutualInformation(TrueY,PredictY)
	H_c = Entropy(PredictY)
	H_t = Entropy(TrueY)
	
	return I/np.sqrt(H_c*H_t)

	

def myAns(clustering_1,clustering_2,clustering_3,clustering_4,clustering_5,partitions):
	from sklearn.metrics.cluster import normalized_mutual_info_score
	import sklearn.metrics
	import numpy as np
	from sklearn.metrics import jaccard_similarity_score
	import numpy as np
	Ans=np.zeros((5,2))
	
	
	Ans[0,0]=normalizedMutualInformation(partitions,clustering_1)
	Ans[0,1]=Jaccard(partitions,clustering_1)

	Ans[1,0]=normalizedMutualInformation(partitions,clustering_2)
	Ans[1,1]=Jaccard(partitions,clustering_2)

	Ans[2,0]=normalizedMutualInformation(partitions,clustering_3)
	Ans[2,1]=Jaccard(partitions,clustering_3)

	Ans[3,0]=normalizedMutualInformation(partitions,clustering_4)
	Ans[3,1]=Jaccard(partitions,clustering_4)

	Ans[4,0]=normalizedMutualInformation(partitions,clustering_5)
	Ans[4,1]=Jaccard(partitions,clustering_5)
	return Ans.tolist()


if __name__ == '__main__':
	import numpy as np
	clustering_1=readFile('./data/clustering_1.txt')
	clustering_2=readFile('./data/clustering_2.txt')
	clustering_3=readFile('./data/clustering_3.txt')
	clustering_4=readFile('./data/clustering_4.txt')
	clustering_5=readFile('./data/clustering_5.txt')
	partitions=readFile('./data/partitions.txt')

	#Ans=SkLearn(clustering_1,clustering_2,clustering_3,clustering_4,clustering_5,partitions)
	Ans=myAns(clustering_1,clustering_2,clustering_3,clustering_4,clustering_5,partitions)
	writeFile('./scores.txt',Ans)
	print(Ans)

	
	print(Jaccard(partitions,clustering_1))
