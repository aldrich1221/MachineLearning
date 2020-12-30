def generateInitialCandidateSet(dataSet):
	candidateSet=[]
	for subset in dataSet:
		for item in subset:
			if not item in candidateSet:
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
	for tid in dataSet:
		for canSet in map(frozenset,candidateSet):
			if canSet.issubset(tid):
				if canSet not in temp:
					temp[canSet]=1
				else:
					temp[canSet]+=1
	for key in temp:
		support=temp[key]/numberOfItems
		if support >=minSupport:
			validList.insert(0,key)
		supportNumber[key]=support
	return validList,supportNumber


def apriori(dataSet,minSupport):
	cand1=generateInitialCandidateSet(dataSet)
	dataSetD=list(map(set,dataSet))
	validList1,supportData=calculateSupport(dataSetD,cand1,minSupport)
	validList=[validList1]
	k=2
	while (len(validList[k-2])>0):
		candidateSet=generateCandidateSet(validList[k-2],k)
		validList_k,supportNumber=calculateSupport(dataSetD,candidateSet,minSupport)
		supportData.update(supportNumber)
		if len(validList)==0:
			break

		validList.append(validList_k)
		k+=1
	return validList,supportData


if __name__ == '__main__':
	#dataSet=[[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
	dataSet=[['A','C','D'],['B','C','E'],['A','B','C','E'],['B','E']]
	list1,supportData=apriori(dataSet,minSupport=0.7)
	print(list1)
	print(supportData)


