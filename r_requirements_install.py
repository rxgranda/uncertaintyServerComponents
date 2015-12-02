#!/usr/bin/python

from rpy2.robjects.packages import importr

R_PACKAGES = [ 'e1071',
               'psych',
               'nFactors',
               'stats',
               'verification', ]

if __name__ == '__main__':
    utils = importr('utils')
    
    for package_name in R_PACKAGES:
        utils.install_packages(package_name)
        
