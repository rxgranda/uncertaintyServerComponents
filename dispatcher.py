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

from fe_process import *
from json import dumps as json_dumps
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
    
    structures_d = {'_programs':_PROGRAM,
                    'core_courses':_COURSES,
                    'conval_dict':_CONVALD,
                    'factors_dict':_FACTORS,
                    'population_IDs':_STUDENTS,
                    }    
    return structures_d
    
class WSDispatcher():

    _structures = None
    
    """
    """
    def __init__(self):
        self._structures = get_structures()
        
        self.academic_clusterer = AcademicClusterer( self._structures['core_courses'],
                                                     self._structures['conval_dict'],
                                                     self._structures['factors_dict'], 
                                                     self._structures['_programs'] )
        self._start_year = -1 
        self._end_year = 1
        
        self.init_estimator()
        
    def init_estimator(self):
        self.academic_estimator = AcademicFailureEstimator(self.academic_clusterer)
        self.academic_estimator.init_semesters_classifier_fn()
        self.academic_estimator.init_students_classifier_fn()
        
    @property
    def structures(self):
        return self._structures
    
    """
    """
    @property
    def students(self):
        return json_dumps( list( self.structures['population_IDs'] ) )
    
    """
    """
    def risk(self, json_input):
        try:
            _root = json_input
            semester = [course['id']  for course in _root['courses']]
            student_ID = int( _root['student'][0]['id'] )
            
            start_year = _root['data'][0]['from']
            end_year = _root['data'][0]['to']
            if self._start_year != start_year or self._end_year != end_year:
                self._start_year = start_year
                self._end_year = end_year
                self.academic_clusterer.set_ha_df(start_year=start_year, end_year=end_year)
                self.init_estimator()
                #print(self.academic_clusterer.rates)
                
            
            risk, quality = self.academic_estimator.predict( student_ID=student_ID,
                                                             semester=semester
                                                             )
        except:
            import traceback
            traceback.print_exc()
            risk, quality = (0,0)
        return json_dumps( {'risk':risk,'quality':quality} )
        
