import pandas as pd
import mysql.connector
import time

mysql_cn= mysql.connector.connect(host='200.10.150.126', 
   port=3306,
   user='utea',
   passwd='t21_sp2', 
   db='csi_extract')

#mysql_cn= mysql.connector.connect(host='200.126.23.31', 
#   port=3356,
#   user='utea',
#   passwd='t21_sp2', 
#   db='csi_extract')

query = '''
SELECT DISTINCT cod_estudiante,promedio,cod_materia_acad,anio,termino,paralelo,estado_mat_tomada
FROM hist_acad_nor
WHERE NOT ( estado_mat_tomada='PE' OR estado_mat_tomada='EX' OR estado_mat_tomada='CV' OR (estado_mat_tomada='AP' AND promedio=0) OR cod_materia_acad LIKE '%NIVEL %' );
'''

start = time.time()
ha_df = pd.read_sql_query(query, con=mysql_cn)
end = time.time()
print "Exe time", end - start
print 'loaded dataframe from MySQL as DataFrame. records:', len(ha_df)
print '\n'
ha_df.to_csv('./data/ha_df.csv', encoding='utf-8')

query = '''
SELECT DISTINCT cod_estudiante,promedio,cod_materia_acad,anio,termino,paralelo,estado_mat_tomada
FROM hist_acad_nor
WHERE ( cod_estudiante IN (SELECT DISTINCT cod_estudiante FROM estudiante_carrera WHERE cod_carrera='ICC' OR cod_carrera='CMP' OR (cod_carrera='ELE' AND cod_especializ='CO')) )
AND NOT ( estado_mat_tomada='PE' OR estado_mat_tomada='EX' OR estado_mat_tomada='CV' OR (estado_mat_tomada='AP' AND promedio=0) OR cod_materia_acad LIKE '%NIVEL %' );
'''

start = time.time()
cp_df = pd.read_sql_query(query, con=mysql_cn)
end = time.time()
print "Exe time", end - start
print 'loaded dataframe from MySQL as DataFrame. records:', len(cp_df)
print '\n'
cp_df.to_csv('./data/cp_df.csv', encoding='utf-8')

query = '''
SELECT DISTINCT cod_estudiante, cod_carrera, cod_especializ, cod_division
FROM hist_acad_nor;
'''
start = time.time()
co_df = pd.read_sql_query(query, con=mysql_cn)
end = time.time()
print "Exe time", end - start
print 'loaded dataframe from MySQL as DataFrame. records:', len(co_df)
print '\n'
co_df.to_csv('./data/co_df.csv', encoding='utf-8')

query = '''
SELECT nombre_materia, cod_materia_acad
FROM materia;
'''

start = time.time()
cs_df = pd.read_sql_query(query, con=mysql_cn)
end = time.time()
print "Exe time", end - start
print 'loaded dataframe from MySQL as DataFrame. records:', len(cp_df)
print '\n'
cs_df.to_csv('./data/cs_df.csv', encoding='utf-8')

query = '''
SELECT cod_estudiante, anio_ingreso, termino_ingreso, cod_nacionalid, sexo, estado_civil, fecha_nacimiento, cod_ciudad_resid, tipodiscapacidad
FROM estudiante;
'''

start = time.time()
pi_df = pd.read_sql_query(query, con=mysql_cn)
end = time.time()
print "Exe time", end - start
print 'loaded dataframe from MySQL as DataFrame. records:', len(pi_df)
print '\n'
pi_df.to_csv('./data/pi_df.csv', encoding='utf-8')

query = '''
SELECT tipoidentif, numeroidentif, anio, termino, cod_materia_acad, paralelo
FROM mat_profesor;
'''

start = time.time()
pc_df = pd.read_sql_query(query, con=mysql_cn)
end = time.time()
print "Exe time", end - start
print 'loaded dataframe from MySQL as DataFrame. records:', len(pc_df)
print '\n'
pc_df.to_csv('./data/pc_df.csv', encoding='utf-8')

mysql_cn.close()
