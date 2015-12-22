#-*- coding: utf-8 -*-

###############################################################################
# MIT License (MIT)
#
# Copyright (c)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
###############################################################################

import pandas as pd
from time import time
from pandas import read_csv as pd_read_csv
from numpy import int32, float32

pd.options.mode.chained_assignment = None
pd.options.mode.use_inf_as_null = True

#def side_strip(_str):
#    try:
#        return _str[ :_str.index(' ') ]
#    except:
#        return _str

'''
Students Academic History

<class 'pandas.core.frame.DataFrame'>
Int64Index: 120080 entries, 0 to 120079
Data columns (total 7 columns):
student        120080 non-null int32
grade          120063 non-null float32
course         120080 non-null object
name           120080 non-null object
status         120080 non-null object
performance    120080 non-null float32
year           120080 non-null int32
dtypes: float32(2), int32(2), object(3)
'''
start = time()
ha_df = pd_read_csv('./data/kuleuven/students_courses.csv')
#ha_df['cod_materia_acad'] = ha_df['cod_materia_acad'].apply( side_strip )
try:
    ha_df['student'] = ha_df['student'].values.astype(int32)
    ha_df['grade'] = ha_df['grade'].values.astype(float32)
    ha_df['performance'] = ha_df['performance'].values.astype(float32)
    ha_df['year'] = ha_df['year'].values.astype(int32)
except:
    pass
end = time()
print('Exe time: %.2f'%(end - start))
print('loaded dataframe from CSV as DataFrame. records: %d'%len(ha_df))
print('\n')

#'''
#Program ID's by Student
#'''
#start = time()
#co_df = pd_read_csv('./data/co_df.csv', index_col=0, dtype={'cod_estduiante':int32})
#end = time()
#print('Exe time: %.2f'%(end - start))
#print('loaded dataframe from CSV as DataFrame. records: %d'%len(co_df))
#print('\n')

'''
Computer Science Students
'''
start = time()
cp_df = pd_read_csv('./data/kuleuven/students.csv', dtype={'student':int32})
end = time()
print('Exe time: %.2f'%(end - start))
print('loaded dataframe from CSV as DataFrame. records: %d'%len(cp_df))
print('\n')

'''
Courses
'''
start = time()
cs_df = pd_read_csv('./data/kuleuven/courses.csv')
#cs_df['cod_materia_acad'] = cs_df['cod_materia_acad'].apply( side_strip )
end = time()
print('Exe time: %.2f'%(end - start))
print('loaded dataframe from CSV as DataFrame. records: %d'%len(cs_df))
print('\n')

#'''
#Students Info
#'''
#start = time()
#pi_df = pd_read_csv('./data/pi_df.csv', index_col=0, dtype={'cod_estduiante':int32})
#end = time()
#print('Exe time: %.2f'%(end - start))
#print('loaded dataframe from CSV as DataFrame. records: %d'%len(pi_df))
#print('\n')

#'''
#Profesor and Courses Links
#'''
#start = time()
#pc_df = pd_read_csv('./data/pc_df.csv', index_col=0)
#pc_df['cod_materia_acad'] = pc_df['cod_materia_acad'].apply( side_strip )
#end = time()
#print('Exe time: %.2f'%(end - start))
#print('loaded dataframe from CSV as DataFrame. records: %d'%len(pc_df))
#print('\n')

