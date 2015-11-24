import pandas as pd
import time
from numpy import int32, float32

pd.options.mode.chained_assignment = None
pd.options.mode.use_inf_as_null = True

def side_strip(_str):
    try:
        return _str[ :_str.index(' ') ]
    except:
        return _str

'''
Students Academic History
'''
start = time.time()
ha_df = pd.read_csv('./data/ha_df.csv', index_col=0)
ha_df['cod_materia_acad'] = ha_df['cod_materia_acad'].apply( side_strip )
try:
    ha_df['cod_estudiante'] = ha_df['cod_estudiante'].values.astype(int32)
    ha_df['promedio'] = ha_df['promedio'].values.astype(float32)
    ha_df['anio'] = ha_df['anio'].values.astype(int32)
    ha_df['paralelo'] = ha_df['paralelo'].values.astype(int32)
    ha_df['GPA'] = ha_df['promedio'].values.astype(float32)
    ha_df['ap_GPA'] = ha_df['promedio'].values.astype(float32)
    ha_df['performance'] = ha_df['promedio'].values.astype(float32)
    ha_df['promedio_GPA'] = ha_df['promedio'].values.astype(float32)
except:
    pass
end = time.time()
print "Exe time",end - start
print 'loaded dataframe from CSV as DataFrame. records:', len(ha_df)
print '\n'

'''
Program ID's by Student
'''
start = time.time()
co_df = pd.read_csv('./data/co_df.csv', index_col=0, dtype={'cod_estduiante':int32})
end = time.time()
print "Exe time",end - start
print 'loaded dataframe from CSV as DataFrame. records:', len(co_df)
print '\n'


'''
Computer Science Students
'''
start = time.time()
cp_df = pd.read_csv('./data/cp_df.csv', index_col=0, dtype={'cod_estduiante':int32})
end = time.time()
print "Exe time",end - start
print 'loaded dataframe from CSV as DataFrame. records:', len(cp_df)
print '\n'

'''
Courses
'''
start = time.time()
cs_df = pd.read_csv('./data/cs_df.csv', index_col=0)
cs_df['cod_materia_acad'] = cs_df['cod_materia_acad'].apply( side_strip )
end = time.time()
print "Exe time",end - start
print 'loaded dataframe from CSV as DataFrame. records:', len(cs_df)
print '\n'

'''
Students Info
'''
start = time.time()
pi_df = pd.read_csv('./data/pi_df.csv', index_col=0, dtype={'cod_estduiante':int32})
end = time.time()
print "Exe time",end - start
print 'loaded dataframe from CSV as DataFrame. records:', len(pi_df)
print '\n'

'''
Profesor and Courses Links
'''
start = time.time()
pc_df = pd.read_csv('./data/pc_df.csv', index_col=0)
pc_df['cod_materia_acad'] = pc_df['cod_materia_acad'].apply( side_strip )
end = time.time()
print "Exe time",end - start
print 'loaded dataframe from CSV as DataFrame. records:', len(pc_df)
print '\n'
