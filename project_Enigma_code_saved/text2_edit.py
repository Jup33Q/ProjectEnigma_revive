import numpy as np
import time

def arr2tab(arr,tablename):
	table=op(tablename)
	row=table.par.rows
	col=table.par.cols

	for i in range(row):
		for j in range(col):
			table[i,j]=arr[i,j]


arr=np.zeros((4,4))

arr2tab(arr,'table1')
