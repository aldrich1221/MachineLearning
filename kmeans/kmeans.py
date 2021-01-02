def readFile(fileName):
	f = open(fileName, mode='r')
	datatxt=[]
	for line in f.readlines():
		number=line.replace('\n','').split(',')

		datatxt.append([float(number[0]),float(number[1])])
	
	f.close
	return datatxt

def Kmeans(k,data, clusterX, clusterY,iterate):
	def distanceFunc(dataX,dataY,clusterX,clusterY):
		return float(((clusterX-dataX)**2 + (clusterY-dataY)**2)**0.5)
	def clustering(X,Y,clusterX,clusterY):
		import sys
		import numpy as np
		dataNum=len(X)
		#print(clusterX,clusterY)
		clusterNum=len(clusterX)
		groups=[]
		Ans=np.zeros(dataNum)
		for i in range(clusterNum):
			groups.append([])
		dis=sys.maxsize
		totalDistance=0
		for idata in range(dataNum):
			for icluster in range(clusterNum):
				distance=distanceFunc(X[idata],Y[idata],clusterX[icluster],clusterY[icluster])
				if distance<dis:
					dis=distance
					chooseGroup=icluster
			groups[chooseGroup].append([X[idata],Y[idata]])

			Ans[idata]=chooseGroup
			totalDistance+=dis
			dis=sys.maxsize
		#print("Average Distance to Cluster Center ",totalDistance/dataNum)

		print(groups[0])
		
		return groups,Ans
	def genearateNewCluster(groups,clusterX,clusterY):
		sumx = 0
		sumy = 0
		new_seed = []
		
		for index, nodes in enumerate(groups):
			
			if nodes == []:
				new_seed.append([clusterX[index], clusterY[index]])
			else:
				for node in nodes:
					sumx += node[0]
					sumy += node[1]
				print(index,float(sumx/len(nodes)),float(sumy/len(nodes)))
				new_seed.append([float(sumx/len(nodes)), float(sumy/len(nodes))])
	        sumx = 0
	        sumy = 0
		newclusterX = []
		newclusterY = []
		for i in new_seed:
			newclusterX.append(i[0])
			newclusterY.append(i[1])
		return newclusterX, newclusterY
	dataX=[]
	dataY=[]
	
	for datapoint in data:
		
		dataX.append(datapoint[0])
		dataY.append(datapoint[1])
	team ,Ans= clustering(dataX, dataY, clusterX, clusterY)
	
	newclusterX, newclusterY = genearateNewCluster(team, clusterX, clusterY)
	print("iter",iterate)
	if iterate>=500:
		return
	if newclusterX==list(clusterX) and newclusterY==list(clusterY):
		return team,Ans
	else:
		iterate+=1
		Kmeans(k,data,clusterX,clusterY,iterate)
		return team,Ans
def initialCenter(data,k):
	import numpy as np
	import random
	centerX=[]
	centerY=[]
	for i in range(k):
		index=random.randint(0, len(data))
		
		centerX.append(data[index][0])
		centerY.append(data[index][1])
	
	
	return centerX,centerY
def writeFile(filename,textdata):
	StringList=[]
	count=0
	for ans in textdata:
		StringList.append(str(count)+" "+str(int(ans))+"\n")
		count=count+1
	f=open(filename,mode='w')
	
	f.writelines(StringList)
	f.close()

def SkLearn(data,k):
	from sklearn.cluster import KMeans
	import numpy as np
	X = data
	kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
	print(kmeans.cluster_centers_)
	return kmeans.labels_

if __name__ == '__main__':
	import numpy as np
	data=readFile('./places.txt')
	
	clusterNum=3
	iterate=0
	initialCenterX,initialCenterY=initialCenter(data,clusterNum)
	team,Ans=Kmeans(clusterNum,data,initialCenterX,initialCenterY,iterate)
	Check=[]
	skans=SkLearn(data,clusterNum)
	for i in range(len(data)):
		print([data[i][0],data[i][1],Ans[i],skans[i]])
		Check.append([data[i][0],data[i][1],Ans[i],skans[i]])
	#print(Check)
	

	writeFile('./clusters.txt',skans)
