from clusterer import *
from classifier_estimator import *
from fe_test import *

academic_clusterer = AcademicClusterer( cs_courses, cs_convald, cs_factors, cs_program )

academic_estimator = AcademicFailureEstimator(academic_clusterer)

academic_estimator.init_semesters_classifier_fn()
academic_estimator.init_students_classifier_fn()

academic_estimator.predict(student_ID=200909893, semester=['ICM00216','ICF01099','ICHE00877','FIEC06460'])
academic_estimator.predict(student_ID=200920254, semester=['ICF01099','ICHE00877','FIEC06460'])
