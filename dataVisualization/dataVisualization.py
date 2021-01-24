if __name__ == '__main__':
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot  as plt
	import matplotlib.ticker as ticker
	Data1=pd.read_csv('./ExcelFormattedGISTEMPDataCSV.csv')
	Data2=pd.read_csv('./ExcelFormattedGISTEMPData2CSV.csv')
	
	
	NewDataFrame=pd.DataFrame()
	def mytest(x):
		if type(x)==int or type(x)==float:
			return True
		else:
			return False


	
	def dataFrameProcess(dataframe):

		for colume in dataframe.columns:
			boolIndex=pd.to_numeric(Data1[colume], errors='coerce').isnull()
		

			dataframe.loc[~boolIndex,colume]=pd.to_numeric(dataframe.loc[~boolIndex,colume])

			#method1 Use mean
			#dataframe.loc[pd.to_numeric(dataframe[colume], errors='coerce').isnull(),colume]=dataframe.loc[~boolIndex,colume].mean()

			#Method2 filled by nan
			dataframe.loc[pd.to_numeric(dataframe[colume], errors='coerce').isnull(),colume]=np.nan
			
		dataframe=dataframe.interpolate(method='linear', limit_direction='forward', axis=0)
		dataframe=dataframe.dropna()
		return dataframe
	

	Data1=dataFrameProcess(Data1)
	print(Data1.head(5))        
	print(Data1.tail(5))
	

	
	plt.style.use("ggplot")       
	
	plt.plot(Data1['Year'], Data1["DJF"],c = "r")  
	
	plt.plot(Data1['Year'], Data1["MAM"], "g-.")

	plt.plot(Data1['Year'], Data1["JJA"], "b-.")

	plt.plot(Data1['Year'], Data1["SON"], "y-.")

	myX_tick=np.array(Data1['Year'])
	
	plt.legend(labels=["DJF", "MAM",'JJA',"SON"], loc = 'best')
	plt.xlabel("Year", fontweight = "bold")              
	plt.ylabel("temperature", fontweight = "bold")    
	plt.title("global temperature trend", fontsize = 15, fontweight = "bold", y = 1.1)   
	plt.xticks(myX_tick[::5])
	plt.xticks(rotation=45)   
	plt.savefig("globalTemperatureTrends.jpg",   
	            bbox_inches='tight',               
	            pad_inches=0.0)                    
	plt.show()
	#plt.close()     
		
		