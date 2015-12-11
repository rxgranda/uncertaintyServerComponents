#!/usr/bin/python

from fe_test import *

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    from uncertaintyServerComponents.clusterer import *
    from ...classifier_estimator.classifier_estimator import *

    academic_clusterer = AcademicClusterer( cs_courses, cs_convald, cs_factors, cs_program )
    academic_estimator = AcademicFailureEstimator(academic_clusterer)
    academic_estimator.init_semesters_classifier_fn()
    academic_estimator.init_students_classifier_fn()
    print( academic_estimator.predict(student_ID=200834711, semester=['ICM00216','ICF01099','ICHE00877','FIEC06460']) )
    print( academic_estimator.predict(student_ID=200920254, semester=['ICF01099','ICHE00877','FIEC06460']) )
#200834711
