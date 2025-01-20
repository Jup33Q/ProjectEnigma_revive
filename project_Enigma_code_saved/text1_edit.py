import numpy as np
import time


def create_table_as_size(rows,cols,tablename):
	table=op(tablename)
	table.par.rows=rows
	table.par.cols=cols



create_table_as_size(4,4,'table1')
