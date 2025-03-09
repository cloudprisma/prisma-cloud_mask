#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import numpy as np
import sys
from time import time
sys.path.append('../../')
import prisma_cloud_mask as prisma
import joblib

import json
import argparse    

def class_balance(df):
    """
    Goals: balance and binarize the target variable "class" in:
    1 (cloud) or 0 (no cloud)
           
    Parameters:
    -----------
        args: df - dataframe with TOA radiance or reflectance values and
        classification for each pixel

    Returns: 
    --------
        balanced_df: balance dataframe

    """     
    
    df['y_b'] = df['label'].astype('int').copy()  
    index = df.query('label != 1').index
    df.loc[index, 'y_b'] = 0
    num_px_cloud = len(df[df['class'] == 'cloud'])

    min_val = min(df['label'].value_counts())
    n_class = len(np.unique(df['label'])) - 1
    min_val = min_val if min_val * n_class < num_px_cloud else num_px_cloud // n_class
    balanced_df = []
    for i in np.unique(df['label']):
        if i != 1:
            balanced_df.append(df.loc[df.label == i].iloc[0:min_val])
        else:
            balanced_df.append(df.loc[df.label == i].iloc[0:min_val*n_class])

    balanced_df = pd.concat(balanced_df)

    return balanced_df


def createParser():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(description='This function contains processes such as binarizing, selecting information. It applies 3 different classifiers to the source dataset ("rf", "xgboost", "knn" ). Also, it returns the trained model in joblib format') 
    parser.add_argument('-i ', '--input', dest='config_getmodel', type=str, required=True, help='Path to the config_getmodel.json format')

    return parser


def cmdLineParse(iargs=None):
    parser = createParser()
    return parser.parse_args(args=iargs)


def apply_model(args=None):
    """
    Goals: apply 3 differents classifiers to the dataset. Uses a df as a 
    source which must be contained in the input configuration file (json) 

    Parameters:
    -----------
        args: absolute path and name for the config_model.json

    Returns:
    -------- 
        joblib files: output file in joblib format as a result of trained model.
        It will be saved in the indicated folder (config_model.json)
    """ 

    inps = cmdLineParse(args)

    if os.path.isfile(inps.config_getmodel):
        with open(inps.config_getmodel, 'r') as f:
            data = json.load(f)
    else:
        print('The configuration file does not exist, check the path file...')
        sys.exit(1)


    model_name = data['model']
    db_folder = data['db_folder']
    file_s = data['csv_fn']
    balance = data['balance']
    out_path_model = data['output_model_folder']
    model_fn = f'{model_name}.joblib'

    if not os.path.isdir(out_path_model): os.mkdir(out_path_model)
    
    df = pd.read_csv(os.path.join(db_folder, file_s))

    # Optional
    # Exluding the cloud shadow class, because the quantity of pixels  
    df = df[df.label != 2]
        
    # Binarizing df source
    print('Pixels per class - without balance:')
    print(df['label'].value_counts())
    if balance:
        df = class_balance(df)
        df['label'].value_counts()
        print('% per class: ')
        print(df['label'].value_counts()/df['label'].count()*100)
        print(df['label'].value_counts())
        print(df.columns)

    # Selecting information
    exclude = ['cloud_mask', 'land_mask', 'geometry', 'image', 'class', 'label', 'y_b']
    X = df[df.columns[~df.columns.isin(exclude)]].copy()
    y = df.y_b
    y = y.astype('category')
   
    X_train, y_train = X, y
            
    # Implementing models 
    # knn:
    a = time()
    if model_name == 'knn':
        model = prisma.knn_classifier.knn_classf(X_train,
                                                 y_train)
    # Rf:                                                                   
    elif model_name == 'rf':
        model = prisma.rf_classifier.rf_classf(X_train,
                                               y_train)
    # XGboost:
    elif model_name == 'xgboost':
        model = prisma.xgboost_classifier.xgboost_classf(X_train,
                                                         y_train)
    print(f'Model: {model_name}\n')
        
    joblib.dump(model, os.path.join(out_path_model, model_fn))
    print(model_name)



if __name__ == '__main__':
    '''
    Main driver.
    '''
    
    apply_model()
