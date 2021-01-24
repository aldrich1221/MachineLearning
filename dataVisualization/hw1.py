if __name__ == '__main__':
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot  as plt
	import matplotlib.ticker as ticker
	Data1=pd.read_csv('./ExcelFormattedGISTEMPDataCSV.csv')
	Data2=pd.read_csv('./ExcelFormattedGISTEMPData2CSV.csv')
	Data1.dropna()
	Data1.drop([0,135])
	Data1=Data1[1:135]
	print(Data1)
	NewDataFrame=pd.DataFrame()
	def mytest(x):
		if type(x)==int or type(x)==float:
			return True
		else:
			return False

	# for i in range(188,202):
	# 	index=Data1(['Year']>i*10&['Year']<(i+1)*10)
	# 	NewDataFrame.append(Data1[index,:].mean())
	# Data1['DJF']=Data1[Data1['DJF'].apply(lambda x: mytest(x))]
	# Data1['MAM']=Data1[Data1['MAM'].apply(lambda x: mytest(x))]
	# Data1['JJA']=Data1[Data1['JJA'].apply(lambda x: mytest(x))]
	# Data1['SON']=Data1[Data1['SON'].apply(lambda x: mytest(x))]
	# print(Data1.head(20))
	# filer=type(Data1['DJF']) !=int
	# print('filer',filer)
	# print(type(Data1['DJF',:]))
	# print(type(Data1['DJF',:])!=int)
	#NewData1=Data1.groupby(Data1.index // 5).agg({'Year':'last','DJF':'mean','MAM':'mean','JJA':'mean','SON':'mean'})
	



	fig = plt.figure()

	ax1 = fig.add_subplot(111)

	



	x=Data1.loc[:,'Year']
	y=Data1.loc[:,'Jan']
	y1=Data1.loc[:,'Feb'] 
	y2=Data1.loc[:,'Mar']  
	y3=Data1.loc[:,'Apr']   
	y4=Data1.loc[:,'May']  
	y5=Data1.loc[:,'Jun']  
	y6=Data1.loc[:,'Jul'] 
	y7=Data1.loc[:,'Aug']
	y8=Data1.loc[:,'Sep']  
	y9=Data1.loc[:,'Oct']  
	y10=Data1.loc[:,'Nov'] 
	y11=Data1.loc[:,'Dec']
	y12=Data1.loc[:,'DJF']
	y13=Data1.loc[:,'MAM']
	y14=Data1.loc[:,'JJA']
	y15=Data1.loc[:,'SON']
	
	

	# ax1.plot(x,y,'-')
	# ax1.plot(x,y1,'-')
	# ax1.plot(x,y2,'-')
	# ax1.plot(x,y3,'-')
	# ax1.plot(x,y4,'-')
	# ax1.plot(x,y5,'-')
	# # ax1.plot(x,y6,'-')
	# ax1.plot(x,y7,'-')
	# ax1.plot(x,y8,'-')
	# ax1.plot(x,y9,'-')
	# ax1.plot(x,y10,'-')
	# ax1.plot(x,y11,'-')
	print(y12.hist())
	print(y13.describe())
	print(y14.describe())
	print(y15.describe())

	plt.figure(figsize = (6, 4.5), dpi = 100)                 
	
	
	line1, = ax1.plot(x, y12, color = 'red', linewidth = 1, label = 'DJF')             
	line2, = ax1.plot(x, y13, color = 'blue', linewidth = 1, label = 'MAM')
	line3, = ax1.plot(x, y14, color = 'green', linewidth = 1, label = 'JJA')             
	line4, = ax1.plot(x, y15, color = 'orange', linewidth = 1, label = 'SON')
	
	tick_spacing = 10
	ax1.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
	
	
	
	ax1.xticks(rotation=90, fontsize=1)
	
	ax1.xlabel('r (m)', fontsize = 1)                        
	ax1.xticks(fontsize = 1)                                 
	ax1.yticks(fontsize = 1)
	ax1.grid(color = 'red', linestyle = '--', linewidth = 1)  
	ax1.ylim(-100, 100)       

	plt.legend(handles = [ line1,line2,line3,line4], loc='upper right')
	
	plt.show()                                                


	# ax1.plot(x,y12,'-')
	
	# ax1.plot(x,y13,'-')
	
	# ax1.plot(x,y14,'-')
	# plt.setp(ax1.get_yticklabels(), visible=False) 
	# ax1.plot(x,y15,'-')
	# plt.setp(ax1.get_yticklabels(), visible=True) 
	# plt.legend()
	# ax1.set_ylabel('Deviation')
	
	# ax1.set_xlabel('Year')
	# ax1.set_title('Temperature Change in different months')
	

	# plt.show()