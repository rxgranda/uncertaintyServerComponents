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

from pandas import DataFrame
from skfuzzy import cmeans_predict
from sklearn.svm import SVC
from numpy import zeros as np_zeros
from numpy import average as np_average

class AcademicFailureEstimator():
    def __init__(self, academic_clusterer, **kwargs):
        """
        """
        if kwargs == {}:
            self._academic_clusterer = academic_clusterer
            self._academic_clusterer.semesters_cluster()
            self._academic_clusterer.students_cluster()
            self._cntr_sf = self._academic_clusterer.cntr_sf
            self._cntr_se = self._academic_clusterer.cntr_se
            self._rates = self._academic_clusterer.rates
    
    """
    """    
    @property
    def students_classifier_fn(self):
        return self._students_clf

    """
    """
    def init_students_classifier_fn(self, **kwargs):
        if kwargs == {}:
            m = 1.25; error = 1.e-10; maxiter = 100
            clf = lambda data: cmeans_predict( data.T, self._cntr_sf, m, error, maxiter )
        else:
            clf = lambda data: cmeans_predict( data.T, self._cntr_sf, kwargs )
        self._students_clf = clf
    
    """
    """
    @property
    def semesters_classifier_fn(self):
        return self._semesters_clf
    
    """
    """
    def init_semesters_classifier_fn(self, **kwargs):
        if kwargs == {}:
            svc = SVC()
        else:
            svc = SVC(kwargs)
        svc.fit( self._cntr_se, range( len( self._cntr_se ) ) )
        svc_predict = svc.predict
        clf = lambda data: svc_predict( data )
        self._semesters_clf = clf       
    
    """
    Use of the certainty value given by Ceratainty=1-Uncertainty
    
    The forecasts are probabilistic, the observations are binary.
    Sample baseline calculated from observations.      

    Brier Score (BS)           =   0.24
    Brier Score - Baseline     =  0.2488
    Skill Score                =  0.03551
    Reliability                =  0.01094
    Resolution                 =  0.01978                       
    Uncertainty              =  0.2488
    """
    def predict(self, student_ID, semester):
        semester_features, student_features = self.get_features( student_ID, semester )
        semester_type = self.semesters_classifier_fn(semester_features)
        #U_, U0_, d_, Jm_, p_, fpc_
        U_, _, _, _, _, fpc_ = self.students_classifier_fn(student_features)
        student_membership = U_.T[0]
        
        set_mask = ( self._rates['km_cluster_ID'] == semester_type[0] )
        print self._rates    
        possibilities = self._rates[ set_mask ]['ratio'].values
        relative_sample_size = self._rates[ set_mask ]['tamanio_relativo'].values
        
        risk = np_average(possibilities, weights=student_membership, axis=0)
        
        certainty = 1. - 0.2488        
        quality = np_average(relative_sample_size, weights=student_membership, axis=0) #+ certainty**2
        if quality > 1:
            quality = 1.
        
        return risk, quality
 
    def get_features(self, student_ID, semester):        
        """
        """
        abs_df = self._academic_clusterer.courses_features
        tmp_df = abs_df[ abs_df[self._academic_clusterer.course_attr].isin(semester) ]
        tse_df = self._academic_clusterer.semesters_features
        tse_df = tse_df[ tse_df[self._academic_clusterer.studentId_attr]==student_ID ]
        if tse_df.empty:
            semester_lvl = 1
        else:
            semester_lvl = tse_df[self._academic_clusterer.SEMESTERS_F_LABELS[0]].values.max() + 1
        alpha = tmp_df['alpha'].values.sum()
        beta = tmp_df['beta'].values.sum()
        skewness = tmp_df['skewness'].values.sum()
        n_courses = len( semester )
        
        semester_features = (semester_lvl, alpha, beta, skewness, n_courses)
        
        cs_df = self._academic_clusterer.students_features
        cs_df = cs_df[ cs_df[self._academic_clusterer.studentId_attr] == student_ID ]
        if cs_df.empty:
            student_features = np_zeros((1,5))
        else:        
            student_features = cs_df[ self._academic_clusterer.STUDENTS_F_LABELS ].values
                
        return semester_features, student_features
        
