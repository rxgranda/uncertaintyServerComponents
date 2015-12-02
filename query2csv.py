#!/usr/bin/python

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

from pandas import read_sql_query
from mysql.connector import connect
from time import time

if __name__ == '__main__':
    mysql_cn= connect( host='200.10.150.126', 
                       port=3306,
                       user='utea',
                       passwd='t21_sp2',
                       db='csi_extract' )

    query = '''
    SELECT DISTINCT cod_estudiante,promedio,cod_materia_acad,anio,termino,paralelo,estado_mat_tomada
    FROM hist_acad_nor
    WHERE NOT ( estado_mat_tomada='PE' OR estado_mat_tomada='EX' OR estado_mat_tomada='CV' OR (estado_mat_tomada='AP' AND promedio=0) OR cod_materia_acad LIKE '%NIVEL %' );
    '''

    start = time()
    ha_df = read_sql_query(query, con=mysql_cn)
    end = time()
    print("Exe time %.2f"%(end - start))
    print('loaded dataframe from MySQL as DataFrame. records: %d'%len(ha_df))
    print('\n')
    ha_df.to_csv('./data/ha_df.csv', encoding='utf-8')

    query = '''
    SELECT DISTINCT cod_estudiante,promedio,cod_materia_acad,anio,termino,paralelo,estado_mat_tomada
    FROM hist_acad_nor
    WHERE ( cod_estudiante IN 
        (SELECT DISTINCT cod_estudiante
         FROM estudiante_carrera 
         WHERE cod_carrera='ICC' OR cod_carrera='CMP' OR (cod_carrera='ELE' AND cod_especializ='CO')) )
    AND NOT ( estado_mat_tomada='PE' OR estado_mat_tomada='EX' OR estado_mat_tomada='CV' OR (estado_mat_tomada='AP' AND promedio=0) OR cod_materia_acad LIKE '%NIVEL %' );
    '''

    start = time()
    cp_df = read_sql_query(query, con=mysql_cn)
    end = time()
    print("Exe time %.2f"%(end - start))
    print('loaded dataframe from MySQL as DataFrame. records: %d'%len(cp_df))
    print('\n')
    cp_df.to_csv('./data/cp_df.csv', encoding='utf-8')

    query = '''
    SELECT DISTINCT cod_estudiante, cod_carrera, cod_especializ, cod_division
    FROM hist_acad_nor;
    '''
    start = time()
    co_df = read_sql_query(query, con=mysql_cn)
    end = time()
    print("Exe time %.2f"%(end - start))
    print('loaded dataframe from MySQL as DataFrame. records: %d'%len(co_df))
    print('\n')
    co_df.to_csv('./data/co_df.csv', encoding='utf-8')

    query = '''
    SELECT nombre_materia, cod_materia_acad
    FROM materia;
    '''

    start = time()
    cs_df = read_sql_query(query, con=mysql_cn)
    end = time()
    print("Exe time %.2f"%(end - start))
    print('loaded dataframe from MySQL as DataFrame. records: %d'%len(cp_df))
    print('\n')
    cs_df.to_csv('./data/cs_df.csv', encoding='utf-8')

    query = '''
    SELECT cod_estudiante, anio_ingreso, termino_ingreso, cod_nacionalid, sexo, estado_civil, fecha_nacimiento, cod_ciudad_resid, tipodiscapacidad
    FROM estudiante;
    '''

    start = time()
    pi_df = read_sql_query(query, con=mysql_cn)
    end = time()
    print("Exe time %.2f"%(end - start))
    print('loaded dataframe from MySQL as DataFrame. records: %d'%len(pi_df))
    print('\n')
    pi_df.to_csv('./data/pi_df.csv', encoding='utf-8')

    query = '''
    SELECT tipoidentif, numeroidentif, anio, termino, cod_materia_acad, paralelo
    FROM mat_profesor;
    '''

    start = time()
    pc_df = read_sql_query(query, con=mysql_cn)
    end = time()
    print("Exe time %.2f"%(end - start))
    print('loaded dataframe from MySQL as DataFrame. records: %d'%len(pc_df))
    print('\n')
    pc_df.to_csv('./data/pc_df.csv', encoding='utf-8')

    mysql_cn.close()
