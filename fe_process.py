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

"""@package docstring
Feature Extraction module.

This module provides students and semesters data as DataFrame structures, academic
and demographic information are used to extract the students and courses features.
"""

import data_loader
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from json import load as json_load
from pandas import DataFrame
from pandas import read_csv as pd_read_csv
from numpy import int32, float32
from rpy2.robjects import FloatVector
from rpy2.robjects.packages import importr
from rpy2.robjects import r
from sklearn.cluster import KMeans as kmeans
from sklearn.metrics import pairwise
from sklearn.preprocessing import scale
from scipy.spatial.distance import cdist

pd.options.mode.use_inf_as_null = True

"""
"""
e1071 = importr('e1071')
psych = importr('psych')
nFactors = importr('nFactors')
stats = importr('stats')

"""
"""
gpa_df = DataFrame()
gpaha_df = DataFrame()
abs_df = DataFrame()
sf_df = DataFrame()
se_df = DataFrame()

"""
"""
_aberrant_value_code = -1000000

def data_structure_from_file(filepath):
    """ 
    Get a JSON file as a python data structure.
    
    The structures soported are list and dict.
    
    Params:
        filepath: The path and name from the file
        
    Returns:
        A dict, list, or nested structure.
    """
    _var = None
    with open(filepath) as infile:
        _var = json_load( infile )
    return _var

def get_ap_mask(cut=6.0):
    """
    Get a Numpy mask to select the approving courses for a student academic history.
    
    Params:
        cut: The cut value for approving.
        
    Returns:
        The numpy mask for select approved courses.
    """
    ap_courses_mask = lambda x: ( ( x['estado_mat_tomada']=='AP' ) \
                                  | ( x['estado_mat_tomada']=='IA' ) \
                                  | ( ( x['estado_mat_tomada']=='IN' ) & ( x['promedio']>=cut ) ) 
                                )
    return ap_courses_mask

def get_rp_mask(cut=6.0):
    """
    Get a Numpy mask to select the reprobating courses for a student academic history.
    
    Params:
        cut: The cut value for approving.
        
    Returns:
        The numpy mask for select reprobated courses.
    """
    rp_courses_mask = lambda x: ( ( x['estado_mat_tomada']=='RP' ) \
                                  | ( x['estado_mat_tomada']=='IR' ) \
                                  | ( ( x['estado_mat_tomada']=='IN' ) & ( x['promedio']<6.0 ) ) \
                                  | ( x['estado_mat_tomada']=='PF' )
                                )
    return rp_courses_mask

def GPA_calc(ha_df):
    ha_gb = ha_df.groupby('cod_estudiante')
    ap_mask = get_ap_mask()
    
    l_gpa_r = []
    
    def GPA_record(academic_history):
        try:
            cod_estudiante = int32( academic_history['cod_estudiante'].values[0] )
        except: 
            cod_estudiante = 0
        _GPA = academic_history['promedio'].values.mean()
        _ap_GPA = academic_history[ ap_mask(academic_history) ]['promedio'].values.mean()
        _performance = ( len( academic_history[ ap_mask(academic_history) ] ) * 1.0 \
        / len( academic_history ) )
        
        tmp = {'cod_estudiante': cod_estudiante,
               'GPA': _GPA,
               'ap_GPA': _ap_GPA,
               'performance': _performance
               }
        return tmp
    
    l_gpa_r = ha_gb.apply( GPA_record )
    gpa_df = DataFrame.from_records( l_gpa_r.tolist() )
    return gpa_df

"""
"""
def get_GPA_by_student(ha_df=data_loader.ha_df):
    global gpa_df
    try:
        if gpa_df.empty:
            #print 'GPA load'
            _gpa_df = pd_read_csv('./data/gpa_df.csv',
            index_col=0,
            dtype={'GPA':float32,
            'ap_GPA':float32,
            'cod_estudiante':int32,
            'performance':float32})
            gpa_df = _gpa_df
        else:
            return gpa_df
    except:
        #print 'GPA load fails'
        _gpa_df = GPA_calc(ha_df)
        _gpa_df.to_csv('./data/gpa_df.csv')
    return _gpa_df

"""
"""
def ah_GPA(ha_df, gpa_df):
    if 'promedio_GPA' in list( ha_df ):
        _gpaha_df = ha_df
    else:
        _gpaha_df = pd_merge(ha_df, gpa_df, on='cod_estudiante', how='left')
        _gpaha_df['promedio_GPA'] = _gpaha_df['promedio'] - _gpaha_df['GPA']
        _gpaha_df.fillna(0.)
        _gpaha_df.to_csv('./data/ha_df.csv', dtype={'cod_estudiante':int32,
        'promedio':float32,
        'anio':int32,
        'paralelo':int32,
        'GPA':float32,
        'ap_GPA':float32,
        'performance':float32,
        'promedio_GPA':float32})
        gpaha_df = _gpaha_df
    return _gpaha_df

"""
""" 
def get_ah(start_year=1959, end_year=2013):
    global gpaha_df
    if gpaha_df.empty:
        ha_df = data_loader.ha_df
        sample_df = ha_df[ ( ha_df['anio'] >= start_year ) & ( ha_df['anio'] <= end_year ) ]
        gpa_df = get_GPA_by_student(sample_df)
        return ah_GPA(sample_df, gpa_df)
    else:
        sample_df = gpaha_df[ ( gpaha_df['anio'] >= start_year ) & ( gpaha_df['anio'] <= end_year ) ]
        return sample_df

"""
""" 
def population_IDs_by_program(co_df, \
                          _programs=[{'cod_carrera':'ELE',\
                                      'cod_especializ':'CO'
                                     }]):
    def get_mask(_programs):
        big_or = None
        for _program in _programs:
            big_and = None
            for _field, _value in _program.iteritems():
                if big_and is None:
                    big_and = ( co_df[_field] == _value )
                else:
                    big_and = big_and & ( co_df[_field] == _value )
            if big_or is None:
                big_or = ( big_and )
            else:
                big_or = big_or | ( big_and )
        return big_or
    
    return np.unique( co_df[ get_mask(_programs) ]['cod_estudiante'].values )

"""
""" 
def get_ahoi(ha_df, core_courses=[], population_IDs=[], mask=None):
    if not mask is None:
        ha_df = ha_df[ mask ]
    if core_courses != []:
        ha_df = ha_df[ ha_df['cod_materia_acad'].isin( core_courses ) ]
    if population_IDs != []:
        ha_df = ha_df[ ha_df['cod_estudiante'].isin( population_IDs ) ]
    return ha_df

"""
""" 
def ah_standardization(ha_df, core_courses, conval_dict, population_IDs=[], mask=None):
    sample_df = get_ahoi( ha_df, population_IDs=population_IDs, mask=mask )
    
    ncc_df = ah_no_core_courses(sample_df, population_IDs=population_IDs, core_courses=core_courses)
    def get_ncc_conval_serie(chunk):
        return chunk['cod_materia_acad'].values.tolist()
        
    def get_ncc_name(clusterID):
        return 'CMP00%02i'%clusterID
    
    def conval_course(course):
        if course in conval_dict.keys():
            #print course
            return conval_dict[course]
        else:
            return course
            
    def ncc_conval_course(course):
        for ncc_code, ncc_cluster in ncc_conval_dict.iteritems():
            if course in ncc_cluster:
                return ncc_code
        return course
    
    def combine_course(chunk):
        for courses in conval_dict['Combine']:
            mask =  ( chunk['cod_materia_acad'].isin( courses ) ) & ( chunk['estado_mat_tomada']=='AP' )
            _comb = chunk[ mask ][ 'promedio' ].values.mean()
            chunk.loc[ mask, 'promedio' ] = _comb
    
    ncc_df['clusterID'] = ncc_df['clusterID'].apply( get_ncc_name )
    ncc_gb = ncc_df.groupby('clusterID')
    ncc_conval_dict = ncc_gb.apply( get_ncc_conval_serie ).to_dict()
    sample_df['cod_materia_acad'] = sample_df['cod_materia_acad'].apply( ncc_conval_course )
    sample_df['cod_materia_acad'] = sample_df['cod_materia_acad'].apply( conval_course )
    
    combine_course( sample_df )
    return sample_df

"""
"""     
def get_standard_ah(core_courses, conval_dict, population_IDs=[], start_year=1959, end_year=2013, program='Computer Science'):
    _h_program = hash( program )
    try:
        sha_df = pd_read_csv( './data/sha_df_%i_%i_%i.csv'%( _h_program, start_year, end_year ))
    except:
        sample_df = get_ahoi( get_ah(start_year=start_year, end_year=end_year),
                              population_IDs=population_IDs )
        sha_df = ah_standardization( sample_df,
                                     core_courses=core_courses,
                                     conval_dict=conval_dict )
        sha_df.to_csv( './data/sha_df_%i_%i_%i'%( _h_program, start_year, end_year ))
    return sha_df

'''
Based on Dunn Index TOP
C	v_Dunn
40	0.702586
5	0.702208
3	0.701602
4	0.701529
25	0.115474
'''
def ah_no_core_courses(ha_df, core_courses=[], population_IDs=[], **kwargs):
    features = ['alpha','beta','skewness','count']
    if kwargs == {}:
        sample_df = get_ahoi(ha_df, population_IDs=population_IDs)
        sample_df = sample_df[ ~sample_df['cod_materia_acad'].isin( core_courses ) ]
        #print sample_df.info()
        abs_df = courses_features_calc( sample_df )
        abs_df.fillna(_aberrant_value_code, inplace=True)
        data = abs_df[features].as_matrix()
        km = kmeans(init='k-means++', n_clusters=40, n_init=10)
        km.fit(data)
        abs_df['clusterID'] = km.labels_
    return abs_df

"""
""" 
def EFA(ha_df, populationIDs=[], core_courses=[], **kwargs):
    sample_df = get_ahoi(ha_df, core_courses, population_IDs)
    sample_df.pivot(index='cod_estudiante', columns='cod_materia_acad', values='promedio')    
    return

"""
""" 
def courses_features_calc(ha_df, population_IDs=[]):
    global e1071
    sample_df = get_ahoi(ha_df, population_IDs=population_IDs)
    skewness = e1071.skewness
    
    def alpha_calc(chunk):
        alpha = ( chunk['GPA'].values**2 ).sum() / ( chunk['promedio'].values * chunk['GPA'].values ).sum()
        return alpha
        
    def beta_calc(chunk):
        beta = ( chunk['promedio_GPA'].values ).sum() / len( chunk )
        return beta

    def skewness_calc(chunk):
        _skewness = skewness( FloatVector( chunk['promedio_GPA'].values ) )
        return _skewness[0]
        
    def count_calc(chunk):
        _count = len( chunk )
        return _count
        
    def course_features_record(academic_history):
        cod_materia_acad = academic_history['cod_materia_acad'].values[0]
        try:
            cod_materia_acad = cod_materia_acad[:cod_materia_acad.index(' ')]
        except: 
            cod_materia_acad = cod_materia_acad
        tmp = {'cod_materia_acad': cod_materia_acad,
               'alpha': alpha_calc( academic_history ),
               'beta': beta_calc( academic_history ),
               'skewness': skewness_calc( academic_history ),
               'count': count_calc( academic_history ),
               }
        return tmp
    
    ha_gb = sample_df.groupby('cod_materia_acad')
    abs_df = ha_gb.apply( course_features_record )
    try: abs_df = DataFrame.from_records( abs_df.tolist() )
    except: pass
    return abs_df

"""
""" 
def alpha_beta_skewness(ha_df, population_IDs=[], program='Computer Science', overwrite=False):
    global abs_df
    _h_program = hash( program )
    
    def calc_n_save():        
        _abs_df = courses_features_calc( ha_df, population_IDs )
        _abs_df.to_csv('./data/abs_df_%i.csv'%( _h_program ))
        return _abs_df

    if abs_df.empty:
        try:
            _abs_df = pd_read_csv('./data/abs_df_%i.csv'%( _h_program ), index_col=0)
            abs_df = _abs_df
        except:
            abs_df = calc_n_save()
    elif overwrite:
        abs_df = calc_n_save()
    return abs_df

"""
""" 
def get_courses_features(ha_df, population_IDs=[]):
    return alpha_beta_skewness( ha_df, population_IDs )

"""
"""
def students_features_calc(ha_df, core_courses, conval_dict, factors_dict, population_IDs=[]):
    sample_df = get_standard_ah( core_courses, conval_dict, population_IDs )
    ap_mask = get_ap_mask()
    
    def get_factors(student_ah):
        #try:
        cod_estudiante = int32( student_ah['cod_estudiante'].values[0] )
        #except:
        #    cod_estudiante = 0
        f_r = {'cod_estudiante':cod_estudiante}
        student_ah_isin = student_ah['cod_materia_acad'].isin
        for factor, courses_set in factors_dict.iteritems():
            set_mask =  student_ah_isin( courses_set )
            student_ah_promedio = student_ah[ set_mask ]['promedio'].values
            f_r[factor] = student_ah_promedio.mean()
            f_r['%s_performance'%factor] = len( student_ah[ ap_mask(student_ah) & set_mask ] ) * 1.0 /\
                                           len( student_ah )
            f_r['%s_measure'%factor] = f_r['%s_performance'%factor] * f_r[factor]
        return f_r
        
    ha_gb = sample_df.groupby('cod_estudiante')
    sf_df = ha_gb.apply( get_factors )
    try:sf_df = DataFrame.from_records( sf_df.tolist() )
    except: pass
    return sf_df

"""
"""
def factors(ha_df, core_courses, conval_dict, factors_dict, population_IDs=[], program='Computer Science'):
    global sf_df
    _h_program = hash( program )
    try:
        if sf_df.empty:
            _sf_df = pd_read_csv( './data/sf_df_%i.csv'%( _h_program ))
            sf_df = _sf_df
            return sf_df
    except:
        _sf_df = students_features_calc( ha_df,
                                         core_courses,
                                         conval_dict,
                                         factors_dict,
                                         population_IDs )
        _sf_df.to_csv('./data/sf_df_%i.csv'%( _h_program ))
        sf_df = _sf_df
    return sf_df

"""
"""
def get_students_features(ha_df, core_courses, conval_dict, factors_dict, population_IDs=[], program='Computer Science'):
    return factors(ha_df, core_courses, conval_dict, factors_dict, population_IDs, program)

"""
"""
def semesters_features_calc(ha_df, core_courses, conval_dict, population_IDs=[]):
    sample_df = get_standard_ah( core_courses, conval_dict, population_IDs )
    sample_df = sample_df.sort_values(['cod_estudiante','anio','termino'])
    abs_df = get_courses_features( sample_df, population_IDs )
    rp_mask = get_rp_mask()
    abs_df_isin = abs_df['cod_materia_acad'].isin
    
    def get_semester_record(chunk):
        taken_courses = chunk['cod_materia_acad'].values
        tmp = { 'cod_estudiante': chunk['cod_estudiante'].values[0],
                'anio': chunk['anio'].values[0],
                'termino': chunk['termino'].values[0],
                'semestre': chunk['semestre'].values[0],
                'materias_tomadas': ' '.join( taken_courses ),
                'materias_reprobadas': ' '.join( chunk[ rp_mask(chunk) ]['cod_materia_acad'].values ),
                'alpha_total': abs_df[ abs_df_isin(taken_courses) ]['alpha'].values.sum(),
                'beta_total': abs_df[ abs_df_isin(taken_courses) ]['beta'].values.sum(),
                'skewness_total': abs_df[ abs_df_isin(taken_courses) ]['skewness'].values.sum(),
                'n_materias': len( abs_df[ abs_df_isin(taken_courses) ] ),
                }
        return tmp
    
    def get_semester_count(chunk):
        i = 0
        chunk_groupby = chunk.groupby
        tc_gb = chunk_groupby(['anio','termino'])
        l_semester_r = []
        extend = l_semester_r.extend
        for _, semester in tc_gb:
            i += 1
            extend( [i] *len(semester) )
        chunk['semestre'] = l_semester_r
        return chunk
        
    cs_gb = sample_df.groupby('cod_estudiante')
    sample_df = cs_gb.apply( get_semester_count )
    ha_gb = sample_df.groupby(['cod_estudiante','anio','termino'])
    se_df = ha_gb.apply( get_semester_record )
    se_df = DataFrame.from_records( se_df.tolist() )
    return se_df

"""
"""
def semesters(ha_df, core_courses, conval_dict, population_IDs=[], program='Computer Science'):
    global se_df
    _h_program = hash( program )
    try:
        if se_df.empty:
            _se_df = pd_read_csv( './data/se_df_%i.csv'%( _h_program ))
            se_df = _se_df
            return se_df
    except:
        _se_df = semesters_features_calc( ha_df,
                                          core_courses,
                                          conval_dict,
                                          population_IDs )
        _se_df.to_csv('./data/se_df_%i.csv'%( _h_program ))
        se_df = _se_df
    return se_df

"""
"""
def get_semesters_features(ha_df, core_courses, conval_dict, population_IDs=[], program='Computer Science'):
    return semesters(ha_df, core_courses, conval_dict, population_IDs, program)

