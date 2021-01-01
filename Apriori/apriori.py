def generateInitialCandidateSet(dataSet):
	candidateSet=[]
	for subset in dataSet:
		for item in subset:
			if not [item] in candidateSet:
				candidateSet.append([item])
	candidateSet.sort()

	return candidateSet
def generateCandidateSet(validList,k):
	candidateSet=[]
	lenValidList=len(validList)
	for i in range(lenValidList):
		for j in range(i+1,lenValidList):
			list1=list(validList[i])[:k-2]
			list2=list(validList[j])[:k-2]
			list1.sort()
			list2.sort()
			if list1==list2:
				candidateSet.append(validList[i]|validList[j])
	
	return candidateSet

	

def calculateSupport(dataSet,candidateSet,minSupport):
	numberOfItems=len(dataSet)
	validList=[]
	supportNumber={}
	temp={}

	for ccanSet in candidateSet:
		for tid in dataSet:
			canSet=frozenset(ccanSet)
			if canSet.issubset(tid):
				if canSet not in temp:
					temp[canSet]=1
				else:
					temp[canSet]+=1
			
	for key in temp:

		supportRatio=float(temp[key])/float(numberOfItems)
		
		if supportRatio >=minSupport:
			validList.append(key)
		supportNumber[key]=temp[key]
		
	return validList,supportNumber

def Check(datatext,item):
	count=0
	for line in datatext:
		if item in line:
			count=count+1
	print("item ", item, "Count",count)
def apriori(dataSet,minSupport,maxLength):
	cand1=generateInitialCandidateSet(dataSet)
	dataSetD=list(map(set,dataSet))
	validList1,supportData=calculateSupport(dataSetD,cand1,minSupport)
	validList=[validList1]
	
	k=2
	
	while (len(validList[k-2])>0):
		if k> maxLength:
			break
		candidateSet=generateCandidateSet(validList[k-2],k)
		
		validList_k,supportNumber=calculateSupport(dataSetD,candidateSet,minSupport)
		
		supportData.update(supportNumber)
		if len(validList_k)==0:
			break

		validList.append(validList_k)
		
		k+=1
	return validList,supportData

def writeFile(filename,textdata):
	StringList=[]
	for key in textdata:
		setString=""
		for element in key:
			setString+=element+";"	
		StringList.append(str(textdata[key])+":"+setString[:-1]+"\n")
	f=open(filename,mode='w')
	
	f.writelines(StringList)
	f.close()
def filterInvalidItem(numberOfItems,minSupport,list1,supportData):
	validAns={}

	for key in supportData:
		
		valid={}
		if supportData[key]>=numberOfItems*minSupport:
			valid[key]=supportData[key]
			validAns.update(valid)
	return validAns
def readFile(fileName):
	f = open(fileName, mode='r')
	datatxt=[]
	for line in f.readlines():
		categories=line.replace('\n','').split(';')
		datatxt.append(categories)
	
	f.close
	return datatxt
if __name__ == '__main__':
	import numpy as np
	import sys
	MAX_INT=sys.maxsize
	
	#datatxt=[['A','C','D'],['B','C','E'],['A','B','C','E'],['B','E']]
	datatxt=readFile("./categories.txt")
	
	Check(datatxt,'Fast Food')
	
	validList,supportData=apriori(datatxt,minSupport=0.01,maxLength=MAX_INT)
	
	finalAns=filterInvalidItem(len(datatxt),0.01,list1,supportData)
	writeFile('./pattern1.txt',finalAns)
	print(list1)
	print(finalAns)


