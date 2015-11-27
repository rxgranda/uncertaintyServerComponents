from fe_process import *
from json import dumps
from numpy.random import uniform
from clusterer import AcademicClusterer
from classifier_estimator import AcademicFailureEstimator

def get_structures( programs_path='./data/_cs_program.txt',
                    core_courses_path='./data/_cs_courses.txt',
                    conval_dict_path='./data/_conval_dict.txt',
                    factors_dict_path='./data/_cs_factors.txt',
                    program='Computer Science' ):
    _PROGRAM = data_structure_from_file(programs_path)
    _COURSES = data_structure_from_file(core_courses_path)
    _CONVALD = data_structure_from_file(conval_dict_path)
    _FACTORS = data_structure_from_file(factors_dict_path)
    _STUDENTS = population_IDs_by_program( data_loader.co_df, _PROGRAM )
    #_RISK = []
    #def risk():
    #for student in _STUDENTS:
    #        risk = {'student_ID':student,'risk':uniform()}
    #        _RISK.append(risk)
    
    structures_d = {'_programs':_PROGRAM,
                    'core_courses':_COURSES,
                    'conval_dict':_CONVALD,
                    'factors_dict':_FACTORS,
                    'population_IDs':_STUDENTS,
                    #'risk':_RISK
                    }    
    return structures_d
    
class WSDispatcher():

    _structures = None
    
    def __init__(self):
        self._structures = get_structures()
        
        self.academic_clusterer = AcademicClusterer( self._structures['core_courses'],
                                                     self._structures['conval_dict'],
                                                     self._structures['factors_dict'], 
                                                     self._structures['_programs'] )
        
        self.academic_estimator = AcademicFailureEstimator(academic_clusterer)
        self.academic_estimator.init_semesters_classifier_fn()
        self.academic_estimator.init_students_classifier_fn()
        
    @property
    def structures(self):
        return self._structures
    
    @property
    def students(self):
        return json.dumps( list( self.structures['population_IDs'] ) )
        
    #@property
    #def risk(self):
    #    return json.dumps( self.structures['risk'] )
    
    def risk(self, json_input):
        _root = json_input[0]
        semester = [course['id']  for course in _root['courses']]
        student_ID = _root['student'][0]['id']
        risk, quality = academic_estimator.predict(student_ID=student_ID, semester=semester)
        return dumps( {'risk':risk,'quality':quality} )
        
