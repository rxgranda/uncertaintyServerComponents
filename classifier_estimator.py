import pandas as pd
from pandas import DataFrame
from skfuzzy import cmeans_predict
from sklearn.svm import SVC
from numpy import average as np_average
from numpy import max as np_max

class AcademicFailureEstimator():
    
    def __init__(self, academic_clusterer, **kwargs):
        if kwargs == {}:
            self._academic_clusterer = academic_clusterer
            self._academic_clusterer.semesters_cluster()
            self._academic_clusterer.students_cluster()
            self._cntr_sf = self._academic_clusterer.cntr_sf
            self._cntr_se = self._academic_clusterer.cntr_se
        self._rt_df = self._academic_clusterer.rates()
        
    @property
    def students_classifier_fn(self):
        return self._students_clf

    def init_students_classifier_fn(self, **kwargs):
        if kwargs == {}:
            m = 1.25; error = 1.e-10; maxiter = 100
            clf = lambda data: cmeans_predict( data.T, self._cntr_sf, m, error, maxiter )
        else:
            clf = lambda data: cmeans_predict( data.T, self._cntr_sf, kwargs )
        self._students_clf = clf
    
    @property
    def semesters_classifier_fn(self):
        return self._semesters_clf
    
    def init_semesters_classifier_fn(self, **kwargs):
        if kwargs == {}:
            svc = SVC()
        else:
            svc = SVC(kwargs)
        svc.fit( self._cntr_se, range( len( self._cntr_se ) ) )
        svc_predict = svc.predict
        clf = lambda data: svc_predict( data )
        self._semesters_clf = clf       
    
    def predict(self, student_ID, semester):
        semester_features, student_features = self.get_features( student_ID, semester )
        semester_type = self.semesters_classifier_fn(semester_features)
        #U_, U0_, d_, Jm_, p_, fpc_
        U_, _, _, _, _, fpc_ = self.students_classifier_fn(student_features)
        student_membership = U_.T[0]
        
        rates = self._academic_clusterer.rates()
        
        set_mask = ( rates['km_cluster_ID'] == semester_type[0] )        
        possibilities = rates[ set_mask ]['ratio'].values
        relative_sample_size = rates[ set_mask ]['tamanio_relativo'].values
        
        risk = np_average(possibilities, weights=student_membership, axis=0)
        quality = np_average(relative_sample_size, weights=student_membership, axis=0)        
        
        return risk, quality
        
        
    def get_features(self, student_ID, semester):
        abs_df = self._academic_clusterer.courses_features
        tmp_df = abs_df[ abs_df['cod_materia_acad'].isin(semester) ]
        tse_df = self._academic_clusterer.semesters_features
        tse_df = tse_df[ tse_df['cod_estudiante']==student_ID ]
        if tse_df.empty:
            semester_lvl = 1
        else:
            semester_lvl = np_max( tse_df['semestre'].values ) + 1
        alpha = tmp_df['alpha'].values.sum()
        beta = tmp_df['beta'].values.sum()
        skewness = tmp_df['skewness'].values.sum()
        n_courses = len( semester )
        
        semester_features = [semester_lvl, alpha, beta, skewness, n_courses]
        
        cs_df = self._academic_clusterer.students_features
        cs_df = cs_df[ cs_df['cod_estudiante'] == student_ID ]
        
        student_features = cs_df[ self._academic_clusterer.STUDENTS_F_LABELS ].values
                
        return semester_features, student_features
        
