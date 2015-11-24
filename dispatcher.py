from fe_process import *
from json import dumps
from numpy.random import uniform

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
    _RISK = []
    #def risk():
    for student in _STUDENTS:
            risk = {'student_ID':student,'risk':uniform()}
            _RISK.append(risk)
    
    structures_d = {'_programs':_PROGRAM,
                    'core_courses':_COURSES,
                    'conval_dict':_CONVALD,
                    'factors_dict':_FACTORS,
                    'population_IDs':_STUDENTS,
                    'risk':_RISK}    
    return structures_d
    
class WSDispatcher():

    _structures = None
        
    @property
    def structures(self):
        if not self._structures:
            self._structures = get_structures()
        return self._structures
    
    @property
    def students(self):
        return json.dumps( list( self.structures['population_IDs'] ) )
        
    @property
    def risk(self):
        return json.dumps( self.structures['risk'] )
