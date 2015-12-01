from fe_process import *

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
#_z = ah_standardization( _tmp_ah, population_IDs=cs_students, core_courses=cs_courses, conval_dict=cs_convald )
#_sf = students_features_calc( _tmp_ah, core_courses=cs_courses, conval_dict=cs_convald, factors_dict=cs_factors, population_IDs=cs_students )

##########################################################################################

"""
_s_df = fe.get_ahoi( _tmp_ah, population_IDs=cs_students)

_ncc_df = ah_no_core_courses(_tmp_ah, population_IDs=cs_students, core_courses=cs_courses)

def get_ncc_conval_serie(chunk):
    return chunk['cod_materia_acad'].values.tolist()
    
def get_ncc_name(clusterID):
    return 'CMP00%02i'%clusterID

_ncc_df['clusterID'] = _ncc_df['clusterID'].apply( get_ncc_name )
_ncc_gb = _ncc_df.groupby('clusterID')
_ncc_conval_dict = _ncc_gb.apply( get_ncc_conval_serie ).to_dict()
"""
#_z = fe.ah_standardization( _tmp_ah, core_courses=cs_courses, population_IDs=cs_students, conval_dict=cs_convald )
#sample_df.pivot(index='cod_estudiante', columns='cod_materia_acad', values='promedio')
#pd.pivot_table(sample_df, index='cod_estudiante', columns='cod_materia_acad', values='promedio')

"""

sample_df = get_standard_ah( cs_courses, cs_convald, cs_students )

In [10]:     ha_gb = sample_df.groupby('cod_estudiante')

In [11]: _x = [ ha_gb.get_group(200909893), ha_gb.get_group(200920254)  ]

    def get_factors(_list, student_ah):
        try:
            cod_estudiante = int32( student_ah['cod_estudiante'].first() )
        except:
            cod_estudiante = 0
        f_r = {'cod_estudiante':cod_estudiante}
        for factor, courses_set in cs_factors.iteritems():
            set_mask =  student_ah['cod_materia_acad'].isin( courses_set ) 
            f_r[factor] = student_ah[ set_mask ]['promedio'].values.mean()
            f_r['%s_performance'%factor] = len( student_ah[ ap_mask(student_ah) & set_mask ] ) * 1.0 /\
                                           len( student_ah )
            f_r['%s_measure'%factor] = f_r['%s_performance'%factor] * f_r[factor]
        l_records.append(f_r)
"""
