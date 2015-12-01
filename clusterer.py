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

import skfuzzy as skf
import pandas as pd
from pandas import DataFrame
from fe_process import *
from sklearn.cluster import KMeans as kmeans
from skfuzzy import cmeans
import warnings

warnings.simplefilter("ignore")
pd.options.mode.use_inf_as_null = True

class AcademicClusterer():
    N = 5
    SEMESTERS_F_LABELS = ['semestre',
                         'alpha_total',
                         'beta_total',
                         'skewness_total',
                         'n_materias']
    STUDENTS_F_LABELS = ['factor%d_measure'%i for i in xrange(1, N+1)]

    def __init__(self, core_courses, conval_dict, factors_dict, _programs, program='Computer Science'):
        self._ha_df = get_ah()
        self._core_courses = core_courses
        self._conval_dict = conval_dict
        self._factors_dict = factors_dict
        self.__programs = _programs
        self._program = program
        
        self._students = None
        self._students_features = None
        self._courses_features = None
        self._semesters_features = None
        self._rates = None
        AcademicClusterer.STUDENTS_F_LABELS = ['%s_measure'%factor for factor in factors_dict.keys()]
    
    """
    """
    @property
    def ha_df(self):
        return self._ha_df
    
    """
    """
    @ha_df.setter
    def ha_df(self, start_year=1959, end_year=2013):
        self._ha_df = get_ah(start_year, end_year)

    """
    """
    @property
    def students(self):
        if self._students is None:
            self._students = population_IDs_by_program( data_loader.co_df,
                                                        self.__programs )
        return self._students
    
    """
    """
    @property
    def students_features(self):
        if self._students_features is None:
            self._students_features = get_students_features( self.ha_df,
                                                             self._core_courses,
                                                             self._conval_dict,
                                                             self._factors_dict,
                                                             self.students,
                                                             self._program )
        return self._students_features
    
    """
    """
    @property
    def courses_features(self):
        if self._courses_features is None:
            self._courses_features = get_courses_features( self.ha_df,
                                                           self.students )
        return self._courses_features
        
    """
    """
    @property
    def semesters_features(self):
        if self._semesters_features is None:
            self._semesters_features = get_semesters_features( self.ha_df,
                                                               self._core_courses,
                                                               self._conval_dict,
                                                               self.students,
                                                               self._program )
            self._semesters_features['materias_reprobadas'] = self._semesters_features['materias_reprobadas'].fillna('')
            self._semesters_features['ha_reprobado'] = self._semesters_features['materias_reprobadas'].apply(lambda x: x=='')
        return self._semesters_features
        
    """
    **********************students_clustering**************************
    Cluster	Ponderation	Pakhira		Partition	Dave	Fukuyama_sugeno
    number	exponent	index		index		index	index
        5	1.250000	0.683793	0.202983	0.860766	-6053.342458
        4	1.250000	1.442550	0.194971	0.852720	-5051.826454
        3	1.250000	1.565918	0.174077	0.848100	-2375.083755
        11	1.250000	0.010866	0.273852	0.848065	-8082.321912
        7	1.250000	0.057873	0.260602	0.841446	-6411.248735
        10	1.250000	0.019743	0.291699	0.834769	-7322.634467
    """ 
    def students_cluster(self, **kwargs):   
        sf_df = self.students_features
        sf_df = sf_df.fillna(0)
        data = sf_df[ self.STUDENTS_F_LABELS ].as_matrix()
        if kwargs == {}:
            C = 5; m = 1.25; error = 1.e-10; maxiter = 100
            cntr, U, _, _, _, _, fpc = cmeans( data.T,
                                               C,
                                               m,
                                               error,
                                               maxiter )
        else:
            cntr, U, _, _, _, _, fpc = cmeans( data, kwargs )
        L = U.T.argmax(axis=1)
        self.students_features['fcm_cluster_ID'] = L
        self.cntr_sf = cntr
    
    """
    ***semesters_clustering***
    N Cluster	Dunn Index
        3		0.997013
        16		0.001198
        19		0.001192
        26		0.000998
        34		0.000958
    """    
    def semesters_cluster(self, **kwargs):
        se_df = self.semesters_features
        se_df.fillna(0)
        data = se_df[ AcademicClusterer.SEMESTERS_F_LABELS ].as_matrix()
        if kwargs == {}:
            C = 3
            km = kmeans(init='k-means++', n_clusters=C, n_init=10)
        else:
            km = kmeans( kwargs )
        km.fit(data)
        cntr, L = km.cluster_centers_, km.labels_
        self.semesters_features['km_cluster_ID'] = L
        self.cntr_se = cntr
        
    """
    """
    @property
    def rates(self):
        def rate_record(chunk):
            tmp = { 'km_cluster_ID': chunk['km_cluster_ID'].values[0],
                    'fcm_cluster_ID': chunk['fcm_cluster_ID'].values[0],
                    'ratio': len( chunk[ chunk['ha_reprobado'] ] ) * 1.0 / len( chunk ),
                    'tamanio': len( chunk ) }
            return tmp
        
        if self._rates is None:
            r_df = pd.merge( self.semesters_features,
                             self.students_features,
                             on='cod_estudiante',
                             how='left' )
        
        
            r_gb = r_df.groupby(['km_cluster_ID','fcm_cluster_ID'])
            df = DataFrame.from_records( list( r_gb.apply( rate_record ) ) )
            df_tamanio_val = df['tamanio'].values
            tamanio_min = df_tamanio_val.min()
            tamanio_max = df_tamanio_val.max()
            df['tamanio_relativo'] = (df_tamanio_val - tamanio_min) * 1.0 / ( tamanio_max - tamanio_min )
            self._rates = df
        return self._rates
