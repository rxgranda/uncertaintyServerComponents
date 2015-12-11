if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from ...fe_process.fe_process import *

    _tmp_ah = get_ah()
    cs_program = data_structure_from_file('./data/_cs_program.txt')
    cs_courses = data_structure_from_file('./data/_cs_courses.txt')
    cs_convald = data_structure_from_file('./data/_conval_dict.txt')
    cs_factors = data_structure_from_file('./data/_cs_factors.txt')
    cs_students = population_IDs_by_program( data_loader.co_df, cs_program )
    _sample_df = _tmp_ah[ _tmp_ah['cod_estudiante'].isin( cs_students ) ]

    __sample_df = _tmp_ah[ ( _tmp_ah['cod_estudiante'].isin( cs_students ) ) &\
        ( _tmp_ah['cod_materia_acad'].isin( cs_courses ) ) &\
        get_ap_mask()( _tmp_ah ) ]
    
    _cs_df = get_ahoi(_sample_df, population_IDs=cs_students)

    #print( _sample_df.info() )
