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
from numpy import zeros as np_zeros
from numpy import ones as np_ones
from numpy import unique as np_unique
from numpy import average as np_average
from numpy import array as np_array
from numpy import append as np_append
from pandas import merge as pd_merge
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC#, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import recall_score
from sklearn.cross_validation import train_test_split
# from sklearn.naive_bayes import GaussianNB
# from sklearn.cluster import KMeans
from itertools import product as it_product

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
            m = self._academic_clusterer._m; error = 1.e-10; maxiter = 100
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
            svc = SVC(kernel='linear', gamma=10e3, probability=True)
        else:
            svc = SVC(kwargs)
        svc.fit( self._cntr_se, range( len( self._cntr_se ) ) )
        svc_predict = svc.predict
        svc_prob = svc.predict_proba
        clf = lambda data: [svc_predict( data ), svc_prob(data)]
        self._semesters_clf = clf

    @staticmethod
    def get_courses_as_bitarray(semester):
        result = \
            [ int( _course in semester ) for _course in AcademicFailureEstimator.COURSES ]
        return result
    
    # @staticmethod
    def get_semester_f(self, semester):
        abs_df = self._academic_clusterer.courses_features
        alpha_total = abs_df[ abs_df['course'].isin(semester) ]['alpha'].sum()
        # credit_total = abs_df[ abs_df['course'].isin(semester) ]['credits'].sum()
        return [alpha_total]
        # return [credit_total]
        # return [alpha_total, credit_total]

    # @staticmethod
    def get_ss_features(self, row):
        semester = row['taken_courses'].split(' ')
        # student_features = [row['GPA'],row['performance_y']]
        student_features = [row['GPA']]
        semester_features = self.get_semester_f(semester)
        return student_features + semester_features

        """KM_FEAT_ = 'factor1_measure', 'factor2_measure', 'factor3_measure',
                'factor4_measure', 'factor5_measure', 'factor6_measure']
        student_features = row[KM_FEAT_].values.tolist()
        semester = row['taken_courses'].split(' ')
        semester_features = [self.get_semester_f(semester)]#get_courses_as_bitarray( row['taken_courses'].split(' ') )
        # print student_features
        # print semester_features
        return student_features + semester_features"""
        """
        KM_FEAT_ = ['factor1_measure', 'factor2_measure', 'factor3_measure',
                    'factor4_measure', 'factor5_measure', 'factor6_measure']
        student_features = row[KM_FEAT_].values.tolist()
        semester = row['taken_courses'].split(' ')
        semester_features = AcademicFailureEstimator.get_courses_as_bitarray( semester )
        return student_features + semester_features"""

    @property
    def classifier_fn(self):
        return self._clf

    """
    """
    def init_classifier_fn(self, **kwargs):
        cs_df = self._academic_clusterer.courses_features
        AcademicFailureEstimator.COURSES = cs_df['course'].values
        
        se_df = self._academic_clusterer.semesters_features
        sf_df = self._academic_clusterer.students_features
        gpa_df = self._academic_clusterer.ha_df.drop_duplicates(['student','GPA'])
        ss_df = pd_merge( se_df, sf_df, on='student' )
        ss_df = pd_merge( ss_df, gpa_df, on='student' )
        ss_df = pd_merge( ss_df, cs_df, on='course' )
        
        data = ss_df.apply( self.get_ss_features, axis=1 )
        data = np_array( data.tolist() )
        X = data
        y = ss_df['ha_reprobado'].apply(lambda x: 0 if x else 1).values

        # H = np_unique( X[:,0] )
        # H = np_array( [ H, np_zeros( len(H) ) ] ).T
        # l = np_ones( len( H ) )
        # X = np_append( X, H, axis=0)
        # y = np_append( y, l )

        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.30,
                                                            random_state=7)

        # logreg = LogisticRegression(random_state=7)
        logreg = AdaBoostClassifier(random_state=10)
        logreg = CalibratedClassifierCV( logreg, cv=2, method='sigmoid')
        # logreg = GaussianNB()
        logreg.fit(X, y)
        logreg_prob = logreg.predict_proba

        y_pred = logreg.predict(X_test)
        recall = recall_score(y_test, y_pred)
        
        clf = lambda data: [ logreg_prob( data ), recall ]
        self._clf = clf
        """
        FEATURES = ['factor1_measure', 'factor2_measure', 'factor3_measure',
                    'factor4_measure', 'factor5_measure', 'factor6_measure',
                    'semester_feature']
                    # 'alpha_total', 'beta_total', 'skewness_total', 'year_n',
                    # 'courses_num']

        AcademicFailureEstimator.COURSES = self._academic_clusterer.courses_features['course'].values
        
        se_df = self._academic_clusterer.semesters_features
        sf_df = self._academic_clusterer.students_features
        ss_df = pd_merge( se_df, sf_df, on='student' )
        ########################################################################
        # ss_df['_class'] = 0

        # mask_c = {}
        # mask_c[0] = ss_df['beta_total']<=-1.5
        # mask_c[1] = (ss_df['beta_total']>-1.5) & (ss_df['beta_total']<-0.862)
        # mask_c[2] = (ss_df['beta_total']>=-0.862) & (ss_df['beta_total']<0)
        # mask_c[3] = ss_df['beta_total']>=0

        # for i in mask_c.keys():
        #     ss_df.loc[ mask_c[i], '_class' ] = i

        tmp = ss_df['taken_courses'].apply(lambda x: x.split())
        Z = tmp.apply( AcademicFailureEstimator.get_courses_as_bitarray )
        Z = np_array( Z.as_matrix().tolist() )

        self._km = KMeans(n_clusters=8, random_state=7)
        _z = self._km.fit_predict( Z )
        
        ss_df['semester_feature'] = _z
        ########################################################################

        KM_FEAT_ = ['factor1_measure', 'factor2_measure', 'factor3_measure',
                    'factor4_measure', 'factor5_measure', 'factor6_measure']
        T = ss_df.drop_duplicates('student')[ KM_FEAT_ ].as_matrix()
        T_ = ss_df[ KM_FEAT_ ].as_matrix()
        
        self._fm = KMeans(n_clusters=4, random_state=7)
        self._fm.fit( T )
        _z = self._fm.predict( T_ )
        
        ss_df['student_feature'] = _z

        ########################################################################

        
        # X = ss_df[ FEATURES ].as_matrix()
        X = ss_df[ ['semester_feature','student_feature'] ].as_matrix()
        y = ss_df['ha_reprobado'].apply(lambda x: 0 if x else 1).values
        # y = ss_df['_class'].values        

        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.30,
                                                            random_state=7)
        
        if kwargs == {}:
            # logreg = SVC(kernel='linear',  tol=0.0001, max_iter=1000, probability=True)
            # logreg = LinearSVC()
            logreg = LogisticRegression(random_state=7)
            # logreg = LogisticRegression(C=1e5)
            # est = GaussianNB()
        else:
            logreg = LogisticRegression(kwargs)
            # est = GaussianNB(kwargs)
        # isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')
        logreg.fit(X, y)
        # isotonic.fit(X, y)
        # logreg_predict = logreg.predict
        # prob_pos = isotonic.predict_proba
        logreg_prob = logreg.predict_proba
        
        ########################################################################
        # prob_pos = logreg.decision_function(X_test)
        # prob_min = prob_pos.min()
        # prob_max = prob_pos.max()
        # prob_den = prob_max - prob_min
        ########################################################################

        # logreg_prob = logreg.decision_function
        
        # y_pred = isotonic.predict(X_test)
        y_pred = logreg.predict(X_test)
        recall = recall_score(y_test, y_pred)
        # clf = lambda data: [ logreg_prob( data ), logreg.score(X, y) ]
        # clf = lambda data: [ (logreg_prob( data ) - prob_min)/prob_den,\
        #                      logreg.score(X, y)]
        clf = lambda data: [ logreg_prob( data ), recall ]
        self._clf = clf
        
        # data_ = [i for i in it_product(range(4),range(8))]        
        # self._clf.ratios = logreg_prob( data_ )
        """
    
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
        if self._academic_clusterer.source == 'espol':
            semester_type = self.semesters_classifier_fn(semester_features)
            #U_, U0_, d_, Jm_, p_, fpc_
            U_, _, _, _, _, fpc_ = self.students_classifier_fn(student_features)
            student_membership = U_.T[0]
            # print(semester_type)
            set_mask = ( self._rates['km_cluster_ID'] == semester_type[0][0] )
            possibilities = self._rates[ set_mask ]['ratio'].values
            relative_sample_size = self._rates[ set_mask ]['tamanio_relativo'].values
            
            risk = np_average(possibilities, weights=student_membership, axis=0)\
                   + semester_type[1][0][semester_type[0][0]]**2
            
            #certainty = 1. - 0.2488        
            quality = np_average(relative_sample_size, weights=student_membership, axis=0) #+ certainty**2
            if quality > 1:
                quality = 1.
            if risk > 1:
                risk = 1.
        elif self._academic_clusterer.source == 'kuleuven':
            # _semester_features = AcademicFailureEstimator.get_courses_as_bitarray( semester )
            gpa_df = self._academic_clusterer.ha_df.drop_duplicates(['student','GPA'])
            _semester_features = self.get_semester_f( semester )
            tmp = gpa_df[ gpa_df['student']==student_ID ]
            # _student_features = [ tmp['GPA'],tmp['performance_y'] ]
            _student_features = [ tmp['GPA'] ]
            # student_semester = student_features[0].tolist() + _semester_features
            student_semester = _student_features + _semester_features

            predict_proba, q = self.classifier_fn( student_semester )
            risk = predict_proba[0][0]
            quality = q
            """
            ####################################################################
            # student_semester = list( semester_features ) + list( student_features[0] )
            semester_features = AcademicFailureEstimator.get_courses_as_bitarray( semester )

            ####################################################################
            
            # student_semester = student_features.tolist()[0].append( self._km.predict( semester_features ) )
            # student_semester = [ self._fm.predict(student_features)[0], self._km.predict( semester_features )[0]]
            
            # print student_semester
            semester_type = self._km.predict( semester_features )[0]

            U_, _, _, _, _, fpc_ = self.students_classifier_fn(student_features)
            student_membership = U_.T[0]

            risk = 0
            for student_type in xrange(4):
                predict_proba, q = self.classifier_fn( [student_type, semester_type] )
                risk += student_membership[student_type]*predict_proba[0][1]
            
            # predict_proba, q = self.classifier_fn( student_semester )
            # risk = predict_proba[0][1]
            # risk = predict_proba[0]
            quality = q
            """
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
        print(semester_features)
        cs_df = self._academic_clusterer.students_features
        cs_df = cs_df[ cs_df[self._academic_clusterer.studentId_attr] == student_ID ]
        if cs_df.empty:
            student_features = np_zeros((1,5))
        else:        
            student_features = cs_df[ self._academic_clusterer.STUDENTS_F_LABELS ].values
                
        return semester_features, student_features
        
