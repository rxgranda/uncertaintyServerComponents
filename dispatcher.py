from fe_process import *


def get_structures(program='Computer Science'):
    _PROGRAM = data_structure_from_file('./data/_cs_program.txt')
    _COURSES = data_structure_from_file('./data/_cs_courses.txt')
    _CONVALD = data_structure_from_file('./data/_conval_dict.txt')
    _FACTORS = data_structure_from_file('./data/_cs_factors.txt')
    _STUDENTS = population_IDs_by_program( data_loader.co_df, cs_program )
    
    structures_d = {'_programs':_PROGRAM,
                    'core_courses':_COURSES,
                    'conval_dict':_CONVALD,
                    'factors_dict':_FACTORS,
                    'population_IDs':_STUDENTS}
    
    return structurs_d
