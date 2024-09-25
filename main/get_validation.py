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
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
import json
import argparse



def createParser():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(description='This function contains the process to get a classification over a whole Prisma image. Save the prediction in a numpy array in npy format. Applies evaluation metrics') 
    parser.add_argument('-i ', '--input', dest='config_getval', type=str, required=True, help='Path to the config_getmodel.json format')

    return parser


def cmdLineParse(iargs=None):
    parser = createParser()
    return parser.parse_args(args=iargs)


def create_validation_ds(img, region, bands, vectors):
		 
    """
    Goals: create a validation data set. Reading an image as a source, 
    it masks radiance data and estimates the reflectance of the data. 
    Creates a filtered dataset using a supervised vectorization.
         
    Parameters:
    -----------
        img: hdf5 file
        region: data region (VNIR / SWIR)
        bands: selected bands
        vectors: source list of vectors related to the database 
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
    
    reflectance[reflectance > 1] = 1
    
    # masking values
    reflectance = prisma.preprocessing.apply_mask(reflectance,
                                                  err_mtx,
                                                  sg_mask)
    
    
    # get geodataframe of radiance per img
    gdf_class = prisma.preprocessing.get_val_geodataframe(reflectance,
                                                          attrs,
                                                          lc_mask,
                                                          cl_mask,
                                                          lat,
                                                          lon,
                                                          vectors)
    
   
    return gdf_class, reflectance, attrs, cl_mask, lat, lon


def classification_val_test(df, model_fn, mask_folder, npy_matrix): 
	
    """
    Goals: predict over the validation dataset. Load a trained class method
    and estimate the prediction over the full image. Save the prediction
    numpy array in npy format.
         
    Parameters:
    -----------
        df: csv file - database of whole image
        model_fn: data region (VNIR / SWIR)
        mask_folder: otuput mask folder

    Returns:
    -------- 
        y_val: true/labeled vector 
        y_pred: prediction vector 
        output_mask: cloud mask classification in numpy array
    """ 
    
    loaded_model = joblib.load(model_fn)
    img_name = df[~df.image.isna()].image.unique()[0]
    df2 = df.copy()
    df2.dropna(inplace=True) 
    idx = np.array(df2.index)
    df2.label = df2.label.astype(int).astype('category')

    exclude = ['cloud_mask', 'land_mask', 'geometry', 'image', 'class', 'label']
    X = df2[df2.columns[~df2.columns.isin(exclude)]].copy()
    y_val = df2.label   
    y_prisma = df2.cloud_mask   

    y_pred = loaded_model.predict(X) # predict_values
    i_roc = []
    
    if len(np.unique(y_val)) > 1:
        i_roc =  loaded_model.predict_proba(X)[:, 1]

    output_mask = np.zeros((1000000)) + 255 
    output_mask[idx] = y_pred  # writing y_pred using idx
    output_mask = output_mask.reshape(1000, 1000)   

    if npy_matrix:
        file_nm = f'{img_name}_{os.path.basename(model_fn).split(".")[0]}_class_val.npy'
        np.save(os.path.join(mask_folder, file_nm), output_mask) 
        
    return y_val, y_pred, y_prisma, output_mask, i_roc

def roc_aucscore(y_val, i_roc):   
    res = round(roc_auc_score(y_val, i_roc), 3) if len(np.unique(y_val)) == 2 else '-'
    return res

def oa(tp, tn, fp, fn):
    res = round((tp + tn)/(tp + tn + fp + fn), 3)
    return res

def ua(tp, fp):
    res = round(tp/(tp+fp), 3) if (tp + fp) != 0 else '-'
    return res

def pa(tp, fn):
    res = round(tp/(tp+fn), 3) if (tp + fn) != 0 else '-'
    return res

def npv(tn, fn):
    res = round(tn/(tn+fn), 3) if (tn + fn) != 0 else '-'
    return res

def sp(tn, fp):
    res = round(tn/(tn+fp), 3) if (tn + fp) != 0 else '-'
    return res
    
def boa(pa, sp):
    res = round((0.5 * pa * sp), 3) if (pa != '-') and (sp != '-') else '-'
    return res
        
def metrics(y_val, y_pred, i_roc, log_fn, t_0, model, source=None):

    """
    Goals: comparison and assessment of class-data. Load validation and prediction 
    vectors. Applies evaluation metrics as: confusion matrix, accuracy,
    precision, recall. Also consider processing time.

    Parameters:
    -----------
        y_val: true/labeled vector 
        y_pred: prediction vector 
        source: source label btw manual or prisma 
        log_fn: logfiles according to 
        t_0: initial processing time
        model: model name (rf, xgboost, knn)
    """ 

    # Create the confusion matrix and metrics
    auc = roc_aucscore(y_val, i_roc) if source[1] == 'prediction' else '-'
    
    cm = confusion_matrix(y_val, y_pred, labels=[1,0])
    tp, fn, fp, tn = cm.ravel()
    
    accuracy = oa(tp, tn, fp, fn)
    UA = ua(tp, fp)
    PA = pa(tp, fn)
    NPV = npv(tn, fn)
    SP = sp(tn, fp)
    BOA =  boa(PA, SP)
    
    with open(log_fn, 'w') as logfile:
        logfile.write(f'Test : {source[0]} - {source[1]}\n')
        logfile.write(f'Model: {model}\n')
        
        logfile.write(f'Confusion matrix: \n {cm.round(4)}\n') 
        logfile.write(f'TN, FP, FN, TP: \n {tn}, {fp}, {fn}, {tp}\n')  
        logfile.write(f'Overall Accuracy - OA: {accuracy}\n')
        logfile.write(f'roc_auc_score: \n {auc}\n')
        logfile.write(f'User Accuracy - UA: {UA}\n')
        logfile.write(f'Producer Accuracy - PA: {PA} \n')
        logfile.write(f'NPV: {NPV} \n')       
        logfile.write(f'SP: {SP} \n')  
        logfile.write(f'Balance between low FP and low FN - BOA: {BOA} \n')           
        logfile.write(f'Processing time: {time() - t_0}s')



def main(args=None):
	
    """
    Goals: This function contains the process to get a classification 
    over a whole Prisma image. Save the prediction in a numpy array in 
    npy or tiff format. Applies evaluation metrics as: confusion matrix, 
    Producer Accuracy (PA), User Accuracy (UA), specificity (SP), negative 
    predictive value (NPV), overall accuracy (OA) and Balanced Overall 
    Accuracy (BOA), and processing time
           
    Parameters
    ----------
        args: absolute path and name for the config_getval.json

    Returns:
    --------   
        Cloud mask classification over a whole Prisma in numpy array

    """ 

    inps = cmdLineParse(args)

    if os.path.isfile(inps.config_getval):
        with open(inps.config_getval, 'r') as f:
            data = json.load(f)
    else:
        print('The configuration file does not exist, check the path file...')
        sys.exit(1)

    path_img = data['img_folder']
    path_vectors = data['val_polygons_folder']
    model_folder = data['trained_models_folder']
    mask_folder = data['output_mask_folder']
    stack_tif = data['stack_tif']
    npy_matrix = data['npy_matrix']
    cloud_prisma = data['cloud_prisma']  
    prisma_test = data['prisma_test']  
    output_log = data['output_log']  

    if not os.path.isdir(mask_folder): os.mkdir(mask_folder)
    if not os.path.isdir(output_log): os.mkdir(output_log)

    model = data['model']
     
    images = [Path(i).as_posix() for i in glob(os.path.join(path_img, '*he5'))]
    vectors = [Path(i).as_posix() for i in glob(os.path.join(path_vectors,'*gpkg'))]
    img_vect_pairs = []
    for i in images:
        for j in vectors:
            img_basename = os.path.basename(i).replace('.he5', '')
            vec_basename = os.path.basename(j).replace('_mk.gpkg', '')
            if img_basename == vec_basename:
                img_vect_pairs.append((i, j))
    
    bands = data['bands']
    region = list(bands.keys())
    region.sort(reverse=True)
    
    print(f'{len(img_vect_pairs)} correspondence image vector found...')
    
    for image, vector in img_vect_pairs:
        name_i = os.path.basename(image).replace('.he5', '')
        print(f'Processing image: {name_i}')
        
        # get geodataframe of radiance per img  
        gdf_class, reflectance, attrs, cl_mask, lat, lon = create_validation_ds(image, region, bands, vector)
        
        t_0 = time()

        # Classification task and get prediction label
        print(f'model: {model}')
        model_name = f'{model}.joblib'
        model_fn = os.path.join(model_folder, model_name) 
        y_val, y_pred, y_prisma, mask, i_roc = classification_val_test(gdf_class, model_fn, mask_folder, npy_matrix)
        
        mask = mask.T
        
        val_fn = f'{name_i}_{model}_val.log'
        log_fn = os.path.join(output_log, val_fn)
        
        # Apply metrics  
        metrics(y_val, y_pred, i_roc, log_fn, t_0, model, source = ['true_label','prediction'])
                
        if prisma_test:
            val_fn = f'{name_i}_val_prisma.log'
            log_fn = os.path.join(output_log, val_fn)  
            metrics(y_val, y_prisma, i_roc, log_fn, t_0, model, source = ['true_label','prisma'])
        
        tiff_msk_fn = f'{name_i}_{model}_class_val.tif'		
        out_tiff = os.path.join(mask_folder, tiff_msk_fn)
        
        # Saving as a Tiff
        mask_bandname = f'cloud_mask_{model}'
              
        if stack_tif:
            bandlist = np.concatenate([attrs[f'List_Cw_{region[0].capitalize()}'], attrs[f'List_Cw_{region[1].capitalize()}']])
            bandlist = [str(i) for i in bandlist] + [mask_bandname]
            ds_out = np.hstack((reflectance, np.expand_dims(mask, axis=1)))
            
            if cloud_prisma:
                bandlist = bandlist + ['cloud_mask_prisma']
                ds_out = np.hstack((ds_out, np.expand_dims(cl_mask, axis=1)))
                tiff_msk_fn = f'{name_i}_{model}_prisma_class_val.tif'
                out_tiff = os.path.join(mask_folder, tiff_msk_fn)
                
        else:
            bandlist = mask_bandname
            ds_out = mask
        
            if cloud_prisma:
                bandlist = [bandlist, 'cloud_mask_prisma']
                cloud_model = np.expand_dims(ds_out, axis=1)
                ds_out = np.hstack((cloud_model, np.expand_dims(cl_mask, axis=1)))
                tiff_msk_fn = f'{name_i}_{model}_prisma_class_val.tif'
                out_tiff = os.path.join(mask_folder, tiff_msk_fn)
        
        print(f'Saving - class image...')  
        prisma.preprocessing.array_totiff(ds_out, lat, lon, out_tiff, bandlist)


if __name__ == '__main__':
	
    '''
    Main driver.
    '''
    main()    

	
