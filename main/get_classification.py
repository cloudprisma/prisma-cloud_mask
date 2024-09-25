#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append('../../')
import prisma_program as prisma
from glob import glob
from pathlib import Path
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from time import time
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import json
import argparse


def createParser():
    '''
    Command line parser.
    '''
    
    parser = argparse.ArgumentParser(description='This function contains the process to get a classification over a whole Prisma image. Save the prediction in a numpy array in npy format. Applies evaluation metrics') 
    parser.add_argument('-i ', '--input', dest='config_getclass', type=str, required=True, help='Path to the config_getclassification.json format')
    
    return parser


def cmdLineParse(iargs=None):
    parser = createParser()
    return parser.parse_args(args=iargs)


def class_ds(img, region, bands):
         
    """
    Goals: reads an image as a source, it masks radiance data and 
    estimates the reflectance of the data. 
             
    Parameters:
    -----------
        img: hdf5 file
        region: data region (VNIR / SWIR)
        bands: selected bands

    Returns:
    --------
        gdf_class: masked geodataframe of a full image in reflectance
    """ 
           
    # load datasets
    arr_vnir, attrs_vnir, err_mtx_vnir, lc_mask, cl_mask, sg_mask, lat, lon = prisma.preprocessing.read_data(img, region[0], bands[region[0]])
    arr_swir, attrs_swir, err_mtx_swir, _, _, _, _, _ = prisma.preprocessing.read_data(img, region[1], bands[region[1]]) # SWIR
       
    # estimate radiance
    arr_rad_vnir = prisma.preprocessing.l1_radiance(arr_vnir, attrs_vnir['ScaleFactor_Vnir'], attrs_vnir['Offset_Vnir'])
    arr_rad_swir = prisma.preprocessing.l1_radiance(arr_swir, attrs_swir['ScaleFactor_Swir'], attrs_swir['Offset_Swir'])
    
    # concatenate radiance
    ds, err_mtx, attrs = prisma.preprocessing.concat_data(arr_rad_vnir,
                                                          arr_rad_swir,
                                                          err_mtx_vnir,
                                                          err_mtx_swir,
                                                          attrs_vnir,
                                                          attrs_swir)

    wl_max = max(bands[region[1]]) if len(bands[region[1]]) > 0 else 2500
    
    reflectance = prisma.preprocessing.l1_reflectance(ds, 
                                                      lat, 
                                                      lon, 
                                                      attrs, 
                                                      wl_max)
    
    # masking values
    reflectance = prisma.preprocessing.apply_mask(reflectance,
                                                  err_mtx,
                                                  sg_mask)
                                                  
    # get geodataframe of radiance per img
    gdf_class = prisma.preprocessing.get_ds(reflectance, 
                                            attrs,
                                            lc_mask,
                                            cl_mask,
                                            lat,
                                            lon)
    
    return gdf_class, reflectance, attrs, cl_mask, lat, lon


def classification_test(df, model_fn, mask_folder, name_i, npy_matrix): 
    
    """
    Goals: predict over a dataset. Load a trained class method
    and estimate the prediction over the full image. Save the prediction
    numpy array in npy format.
         
    Parameters:
    -----------
        df: csv file - database of whole image
        model_fn: data region (VNIR / SWIR)
        mask_folder: otuput mask folder
    
    Returns:
    -------- 
        y_pred: prediction vector 
        output_mask: cloud mask classification in numpy array
    """ 
    
    loaded_model = joblib.load(model_fn)
    
    df2 = df.copy()
    df2.dropna(inplace=True) 
    idx = np.array(df2.index)   
    exclude = ['cloud_mask', 'land_mask', 'geometry'] 
    
    X = df2[df2.columns[~df2.columns.isin(exclude)]].copy()  
    y_pred = loaded_model.predict(X) # predict_values
    output_mask = np.zeros((1000000)) + 255  
    output_mask[idx] = y_pred  # writing y_pred using idx
    output_mask = output_mask.reshape(1000, 1000)   
	
    if npy_matrix:
        file_nm = f'{name_i}_{os.path.basename(model_fn).split(".")[0]}_class.npy'
        np.save(os.path.join(mask_folder, file_nm), output_mask) 
            
    return y_pred, output_mask    
    

def main(args=None):
    
    """
    Goals: This function contains the process to get a classification 
    over a whole Prisma image. Save the prediction in a numpy array in 
    npy or tiff format.
           
    Parameters
    ----------
        args: absolute path and name for the config_getclass.json
    
    Returns:
    --------   
        Cloud mask classification over a whole Prisma in numpy array
    
    """ 
    
    inps = cmdLineParse(args)
    if os.path.isfile(inps.config_getclass):
        with open(inps.config_getclass, 'r') as f:
            data = json.load(f)
    else:
        print('The configuration file does not exist, check the path file...')
        sys.exit(1)
    
    path_img = data['img_folder']
    model_folder = data['trained_models_folder']
    mask_folder = data['output_mask_folder']
    stack_tif = data['stack_tif']
    cloud_prisma = data['cloud_prisma']
    npy_matrix = data['npy_matrix']
    
    if not os.path.isdir(mask_folder): os.mkdir(mask_folder)
    
    model = data['model']
          
    images = [Path(i).as_posix() for i in glob(os.path.join(path_img, '*he5'))]
    bands = data['bands']
    region = list(bands.keys())
    region.sort(reverse=True)
    
    print(f'{len(images)} images found...')    
    for i in range(len(images)):
        image = images[i]
        name_i = os.path.basename(image).rstrip('.he5')
        print(f'Processing image: {name_i}')
        gdf_class, ds, attrs, cl_mask, lat, lon = class_ds(image, region, bands)
        print(f'model: {model}')
        model_name = f'{model}.joblib'
        model_fn = os.path.join(model_folder, model_name) 
        
        print(f'Classifying image: {name_i} ')
        y_pred, mask = classification_test(gdf_class, model_fn, mask_folder, name_i, npy_matrix)
        mask = mask.T
        
        tiff_msk_fn = f'{name_i}_{model}_class.tif'
        out_tiff = os.path.join(mask_folder, tiff_msk_fn)
        
        # saving as a Tiff
        mask_bandname = f'cloud_mask_{model}'
        
      
        if stack_tif:
            bandlist = np.concatenate([attrs[f'List_Cw_{region[0].capitalize()}'], attrs[f'List_Cw_{region[1].capitalize()}']])
            bandlist = [str(i) for i in bandlist] + [mask_bandname]
            ds_out = np.hstack((ds, np.expand_dims(mask, axis=1)))
            if cloud_prisma:
                bandlist = bandlist + ['cloud_mask_prisma']
                ds_out = np.hstack((ds_out, np.expand_dims(cl_mask, axis=1)))
                tiff_msk_fn = f'{name_i}_{model}_prisma_class.tif'
                out_tiff = os.path.join(mask_folder, tiff_msk_fn)
                
        else:
            bandlist = mask_bandname
            ds_out = mask
        
            if cloud_prisma:
                bandlist = [bandlist, 'cloud_mask_prisma']
                cloud_model = np.expand_dims(ds_out, axis=1)
                ds_out = np.hstack((cloud_model, np.expand_dims(cl_mask, axis=1)))
                tiff_msk_fn = f'{name_i}_{model}_prisma_class.tif'
                out_tiff = os.path.join(mask_folder, tiff_msk_fn)
        
        print(f'Saving image...') 
        prisma.preprocessing.array_totiff(ds_out, lat, lon, out_tiff, bandlist)



if __name__ == '__main__':
    
    '''
    Main driver.
    '''
    main()    

    
