#!/usr/bin/python

# wrapper for pre_process_data and explo]
import entropy_measures as em
import pandas as pd
import numpy as np
import gcsfs
import gc

import pdb
import argparse




import pre_process_data as ppd
import exploratory_analysis as ea

'''
This file serve as main wrapper for pre_process_data.py and exploratory_analysis.py
'''
# no command lines
if __name__ == '__main__':
    # select users based on total stream time and filter out by entropy outliers
    df = ppd.prepare_data()

    # create normalization model, PCA model and isolation tree model
    ea.buil_model(df, True)
