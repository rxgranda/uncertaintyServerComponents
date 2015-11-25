import skfuzzy as skf
import pandas as pd
from pd import options.mode as mode
from fe_process import *
from sklearn.cluster import KMeans as kmeans
from skfuzzy import cmeans

mode.use_inf_as_null = True

class StudentClusterer():
    N = 5
    SEMESTERS_F_LABELS = ['semestre',
                         'alpha_total',
                         'beta_total',
                         'skewness_total',
                         'n_materias']
    STUDENTS_F_LABELS = ['factor%d_measure'%i in i in xrange(N)]

    def _init(self, core_courses, conval_dict, factors_dict, _programs, program='Computer Science'):
        self._ha_df = get_ah()
        self._core_courses = core_courses
        self._conva_dict = conval_dict
        self._factors_dict = factors_dict
        self.__programs = _programs
        self._program = program
        StudentClusterer.STUDENTS_F_LABELS = ['%s_measure'%factor for factor in factors_dict.keys()]
        return self
    
    @property
    def ha_df(self):
        return self._ha_df
    
    @ha_df.setter
    def ha_df(self, start_year=1959, end_year=2013):
        self._ha_df = get_ah(start_year, end_year)

    @property
    def students(self):
        self._students = population_IDs_by_program( data_loader.co_df,
                                                    self._program )
        return self._students
    
    @property
    def students_features(self):
        self._students_features = get_students_features( self.ha_df,
                                                         self._core_courses,
                                                         self._conval_dict,
                                                         self._factors_dict,
                                                         self.students,
                                                         self._program )
        return self._students_features
        
    @property
    def semesters_features(self):
        self._semesters_features = get_semesters_features( self.ha_df,
                                                           self._core_courses,
                                                           self._conval_dict,
                                                           self._population_IDs,
                                                           self._program )
        self._semesters_features
        
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
        sf_df.fillna(0)
        data = sf[ StudentClusterer.STUDENTS_F_LABELS ].as_matrix()
        if kwargs == {}:
            C = 5; m = 1.25, error = 1.e-10; masiter = 100
            cntr, U, _, _, _, _, fpc = cmeans(data, C, m, error, maxiter)
        else:
            cntr, U, _, _, _, _, fpc = cmeans(data, kwargs)
        
            

