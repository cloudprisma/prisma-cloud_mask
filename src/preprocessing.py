#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py
import os
import numpy as np
import sys 
from pysolar.solar import get_altitude
from datetime import datetime, timezone, timedelta
from pvlib.solarposition import nrel_earthsun_distance
import dateutil.parser
import bz2

import pandas as pd
import geopandas as gpd



import rasterio
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint as GCP


# ~ np.seterr(divide='ignore', invalid='ignore')


"""
    Note:
    This is adapted to Python from the original code developed by:    
    * ACOLITE: generic atmospheric correction module - for PRISMA 
    - All credit goes to the original author.
    source: https://github.com/acolite/acolite
"""

def get_sun_zenith_angle(lat, lon, t):
    """
    Goals: get the sun zenith angle for all pixels in the scene 
    Parameters
    ----------
        lat: latitude np array
        lon: longitude np array
        t: acquisition time average of the scene
    Returns-> 
    sun zenith angle (degrees)
    """    
    
    sza = float(90) - get_altitude(lat, lon, t)
    return sza


def get_f0(filename):
    """
    Goals: Thuillier transform data in to a dictionary format {"wave": 'data'}
    Parameters
    ----------
        filename: path to Thuillier2003.txt.bz2
    Returns-> 
    Thuillier 2003, in dictionary format 
    
    ----------
    Note:
    This is adapted to Python from the original code developed by:    
    * ACOLITE: generic atmospheric correction module - for PRISMA 
    - All credit goes to the original author.
    source: https://github.com/acolite/acolite
    """        
    f0_list = []
    wl_list = []
    
    if os.path.isfile(filename):
        with bz2.open(filename, 'rb') as fn:
            lines = [l.decode('utf-8') for l in fn.readlines()]
        for line in lines:
            if not line.startswith('#'):
                wl, f0 = line.split()
                wl_list.append(float(wl))
                f0_list.append(float(f0))
        f0_dict = {"wave": np.array(wl_list), "data": np.array(f0_list)}
        return(f0_dict)
        
    else:
        print(f'File {filename} is not in the directory')
        sys.exit(1)


def gauss_response(wl, fwhm, step=0.1):
    """
    Goals: get the Gauss response from wavelength and fwhm information
    Parameters
    ----------
        wl: wavelength - extracted of metadata
        fwhm: full width at half maximum - extracted of metadata
    Returns->
    x, y: values of gauss response.
    
    ----------
    Note:
    This is adapted to Python from the original code developed by:    
    * ACOLITE: generic atmospheric correction module - for PRISMA 
    - All credit goes to the original author.
    source: https://github.com/acolite/acolite
    """        
    
            
    wl_min, wl_max = [wl - 1.5*fwhm, wl + 1.5*fwhm]
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    x = np.linspace(wl_min, wl_max, int(1 + (wl_max - wl_min) / step))
    # ~ breakpoint()
    y = np.exp(-((x - wl) / sigma)**2)
    return(x, y)


def rsr_convolution(wl_data, data, rsr, range_wl=[0.2, 2.55], step=0.001):
    """
    Goals: get the convolution btw spectrums, and get the Solar Irradiance
    Parameters
    ----------
        wl_data: wavelength related to Thuillier spectrum - (dic)
        data: responses related to Thuillier spectrum - (dic)
        rsr: dic of Gaussian wavelength and responses related to Prisma
    Returns-> 
    resdata: dic of solar irradiance
    
    ----------
    Note:
    This is adapted to Python from the original code developed by:    
    * ACOLITE: generic atmospheric correction module - for PRISMA 
    - All credit goes to the original author.
    source: https://github.com/acolite/acolite
    """       
    
    ## set up wavelength space
    interp_wl = np.linspace(range_wl[0],
                            range_wl[1],
                            int(((range_wl[1] - range_wl[0]) / step) + 1))

    ## interpolate RSR to same dimensions
    rsr_interp = dict()
    
    for band in rsr:
        interp_band_value = np.interp(interp_wl,
                                      rsr[band]['wave'],
                                      rsr[band]['response'],
                                      left=0, right=0)
        
        rsr_interp[band] = {'wave': interp_wl,
                            'response': interp_band_value,
                            'sum': interp_band_value.sum()}

    interp_data = np.interp(interp_wl, wl_data, data, left=0, right=0)
    
    resdata = {band: (sum(interp_data * rsr_interp[band]['response'])/rsr_interp[band]['sum']) for band in rsr}
    
    return resdata


def l1_radiance(ds, scale_factor, offset):
    """
    Goals: scaling cube data
    Parameters
    ----------
        ds: cube corresponded to the sensor
        scale_factor: scale factor corresponded to the sensor
        offset: offset corresponded to the sensor
    Returns-> 
    scaled cube
    """
    

    cube = ds/scale_factor - offset
	
    return cube


def l1_reflectance(radiance, lat, lon, attrs, bmax):
    """
    Goals: get reflectance TOA data
           It calls other functions such as: get_sun_zenith_angle,
           gauss_response, get_f0, rsr_convolution, pvlib.solarposition,
           
    Parameters
    ----------
        radiance: radiance dataset
        lat: latitude information
        lon: longitude information
        attrs: metadata
    Returns->   
    ptoa: reflectance TOA np array 
    
    ----------
    Note:
    This is adapted to Python from the original code developed by:    
    * ACOLITE: generic atmospheric correction module - for PRISMA 
    - All credit goes to the original author.
    source: https://github.com/acolite/acolite
    """  
    
    
    
    if bmax > 2400: 
        print('Extraterrestrial solar irradiance - Coddington 2023, will be use as a source...')
        f0_name = '../data/Coddington2023_1_0nm.txt.bz2'
    else:
        f0_name = '../data/Thuillier2003.txt.bz2'
    
    start_dt = dateutil.parser.parse(attrs['Product_StartTime']).replace(tzinfo=timezone.utc)
    stop_dt = dateutil.parser.parse(attrs['Product_StopTime']).replace(tzinfo=timezone.utc)
    mean_dt = start_dt + timedelta(seconds=(stop_dt - start_dt).seconds/2)
    
    sza = get_sun_zenith_angle(lat, lon, mean_dt)  # Solar Zenith Angle
    sza = np.cos(np.deg2rad(sza))  # cosine Solar Zenith Angle
    
    # 2. 
    wl_vnir = attrs['List_Cw_Vnir']
    bands_vnir = [f'{wl:.0f}' for wl in wl_vnir]
    fwhm_vnir = attrs['List_Fwhm_Vnir']
    n_vnir = len(wl_vnir)
    
    wl_swir = attrs['List_Cw_Swir']
    bands_swir = [f'{w:.0f}' for w in wl_swir]
    fwhm_swir = attrs['List_Fwhm_Swir']
    n_swir = len(wl_swir)
    
    wl = list(wl_vnir) + list(wl_swir)
    fwhm = list(fwhm_vnir) + list(fwhm_swir)
    wl_names = bands_vnir + bands_swir
    
    # 4. 
    rsr_vnir = {f'vnir_{b}': gauss_response(wl_vnir[b], fwhm_vnir[b], step=0.1) for b in range(0, n_vnir)}
    rsr_swir = {f'swir_{b}': gauss_response(wl_swir[b], fwhm_swir[b], step=0.1) for b in range(0, n_swir)}
    
    # 5. 
    band_rsr = {}

    for b in rsr_vnir:
        band_rsr[b] = {'wave': rsr_vnir[b][0]/1000, 'response': rsr_vnir[b][1]}

    for b in rsr_swir:
        band_rsr[b] = {'wave': rsr_swir[b][0]/1000, 'response': rsr_swir[b][1]}
    
    # 6. 
    f0 = get_f0(f0_name)
    f0d = rsr_convolution(f0['wave']/1000, f0['data'], band_rsr)
    
    # 8.
    d = nrel_earthsun_distance(mean_dt).iloc[0]
    
    # 9. 
    ptoa = np.zeros_like(radiance)
    
    for pos, esun in enumerate(f0d.values()):
        ptoa[:, pos, :] = (np.pi*d**2 * radiance[:, pos, :]) / (esun * sza)
    
    return ptoa


def read_data(filename, spectral_region, bands=[]):
    """
    Goals: open-read and extract data and metadata of Prisma L1 
    Parameters
    ----------
        filename: path and name of the dataset
        spectral_region: VNIR - SWIR
        bands: list of bands in cube (optional)
    Returns-> 
    data: cube (ds), error matrix (err_mtx),
    masks: landmask, cloudmask, sunglintmask, lat, lon, in np array 
    metadatainfo: attrs
    """

    sensor = 'HCO'
    if not os.path.exists(filename):
        print(f'File {filename}, not found...\n')
        print('Check your path or file')
        sys.exit(1)

    prs_name = f'//HDFEOS/SWATHS/PRS_L1_{sensor}'
    
    with h5py.File(filename,  mode='r') as hf:
        if spectral_region.lower().endswith('_cube'):
            spectral_region = spectral_region[:-5]
        # For longer names:
        if len(spectral_region) == 4:
            sds = f'{prs_name}/Data Fields/{spectral_region.upper()}_Cube'
        elif len(spectral_region) > 4:
            sr = spectral_region.split('_')
            sds = f'{prs_name}/Data Fields/{sr[0].upper()}_{"_".join(sr[1:])}_Cube'
        else:
            print('Select a Data Cube')

        prod_msk_lc = f'{prs_name}/Data Fields/LandCover_Mask'
        prod_msk_cm = f'{prs_name}/Data Fields/Cloud_Mask'
        prod_msk_sm = f'{prs_name}/Data Fields/SunGlint_Mask'
        prod_err = f'{prs_name}/Data Fields/{spectral_region[:4].upper()}_PIXEL_SAT_ERR_MATRIX'

        prod_lat = f'{prs_name}/Geolocation Fields/Latitude_{spectral_region[:4].upper()}'
        prod_lon = f'{prs_name}/Geolocation Fields/Longitude_{spectral_region[:4].upper()}'

        attrs = dict(hf.attrs)
        attrs = {i: attrs[i] for i in attrs.keys() if spectral_region[:4].lower() in i.lower() or i.lower().startswith('product')}
        
        
        landmask = hf[prod_msk_lc][:]
        cloudmask = hf[prod_msk_cm][:]
        sunglintmask = hf[prod_msk_sm][:] 
        lat = hf[prod_lat][:]
        lon = hf[prod_lon][:]
        wavelength = f'List_Cw_{spectral_region[:4].capitalize()}'
        fwhm = f'List_Fwhm_{spectral_region[:4].capitalize()}'
		
        ds = hf[sds][:].astype(np.single)
        err_mtx = hf[prod_err][:]
        
    if len(bands) > 0:
        band_pos = [np.argmin(abs(i - attrs[wavelength])) for i in bands]
        ds = ds[:, band_pos, :]
        attrs[wavelength] = attrs[wavelength][band_pos]
        attrs[fwhm] = attrs[fwhm][band_pos]
        err_mtx = err_mtx[:, band_pos, :]
    
    ds = ds[:, attrs[wavelength] != 0, :]
    err_mtx = err_mtx[:, attrs[wavelength] != 0, :]
    attrs[fwhm] = attrs[fwhm][attrs[wavelength]!=0]
    attrs[wavelength] = attrs[wavelength][attrs[wavelength]!=0]
    
    return ds, attrs, err_mtx, landmask, cloudmask, sunglintmask, lat, lon


def concat_data(arr_vnir, arr_swir, err_mtx_vnir, err_mtx_swir, attrs_vnir, attrs_swir):
    """
    Goals: concatenate information (np array & metadata), related to the sensor.
    Parameters
    ----------
        arr_vnir, arr_swir: cube of scaled data in np array format
        err_mtx_vnir, err_mtx_swir: error matrix of data in np array format
        attrs_vnir, attrs_swir: metadata information
    Returns-> 
    merge of datasets related to the sensor
    """    
    
    ds = np.concatenate([arr_vnir, arr_swir], axis=1)  
    err_mtx = np.concatenate([err_mtx_vnir, err_mtx_swir], axis=1)
    attrs = {**attrs_vnir, **attrs_swir}

    return ds, err_mtx, attrs   


def apply_mask(ds, err_mtx, sg_mask):
    """
    Goals: apply sunglint and error masks to the information
    Parameters
    ----------
        ds: full cube of data (VNIR and SWIR)
        err_mtx : full error matrix of data
        sg_mask: mask of sunglint 
    Returns-> 
    dataset with error/sunglint pixels changed to nan values
    """        
    
    ds[err_mtx != 0] = np.nan  
    mask = sg_mask==0
    mask = np.expand_dims(mask, 1)
    mask_sg = np.repeat(mask, ds.shape[1], axis=1)
    ds[~mask_sg] = np.nan
    return ds

        
def get_geodataframe(ds, attrs, lc_mask, cl_mask, lat, lon, img, vectores):
    """
    Goals: get the database and remove unnecessary information 
    Parameters
    ----------
        ds: data with error/sunglint pixels changed to nan values
        attrs : metadata information
        lc_mask, cl_mask: mask of landmask and cloundmask 
        lat, lon: latitude and longitude values
        img, vectores: source list of images and vectors related to the database
    Returns-> 
    filtered dataset by supervised vectorization 
    """   
    n_bands = [f'{i:.03f}' for i in attrs['List_Cw_Vnir'].tolist() + attrs['List_Cw_Swir'].tolist()]
    names_bands =  n_bands + ['lat', 'lon', 'cloud_mask', 'land_mask'] 
    lat = np.expand_dims(lat, axis=1)
    lon = np.expand_dims(lon, axis=1)
    cl_mask = np.expand_dims(cl_mask, axis=1)
    lc_mask = np.expand_dims(lc_mask, axis=1)
    ds_bands = np.hstack((ds, lat, lon, cl_mask, lc_mask))
    ds_bands = np.transpose(ds_bands, axes=[2,0,1])
     
    df_ds = pd.DataFrame(ds_bands.reshape(-1, ds_bands.shape[2]), columns=names_bands)
    gdf = gpd.GeoDataFrame(df_ds, geometry=gpd.points_from_xy(df_ds['lon'], df_ds['lat'], crs='EPSG:4326')) 
    gdf.drop(columns=['lat', 'lon'], inplace=True)
    vect_name = [i for i in vectores if os.path.basename(img).rstrip('.he5') in i][0]
    supervised_class = gpd.read_file(vect_name) 
    gdf_class = gdf.sjoin(supervised_class)
    gdf_class.drop(columns='index_right', inplace=True)
    
    return gdf_class


def get_ds(ds, attrs, lc_mask, cl_mask, lat, lon):
    """
    Goals: get the database and remove unnecessary information 
    Parameters
    ----------
        ds: data with error/sunglint pixels changed to nan values
        attrs : metadata information
        lc_mask, cl_mask: mask of landmask and cloundmask 
        lat, lon: latitude and longitude values
    Returns-> 
    filtered dataset by supervised vectorization 
    """   
    n_bands = [f'{i:.03f}' for i in attrs['List_Cw_Vnir'].tolist() + attrs['List_Cw_Swir'].tolist()]
    names_bands =  n_bands + ['lat', 'lon', 'cloud_mask', 'land_mask'] 
    lat = np.expand_dims(lat, axis=1)
    lon = np.expand_dims(lon, axis=1)
    cl_mask = np.expand_dims(cl_mask, axis=1)
    lc_mask = np.expand_dims(lc_mask, axis=1)
    ds_bands = np.hstack((ds, lat, lon, cl_mask, lc_mask))
    ds_bands = np.transpose(ds_bands, axes=[2,0,1])
     
    df_ds = pd.DataFrame(ds_bands.reshape(-1, ds_bands.shape[2]), columns=names_bands)
    df_ds = df_ds.loc[:, df_ds.isna().sum() / len(df_ds) < 0.05]  
    df_ds.dropna(inplace=True) 
    gdf = gpd.GeoDataFrame(df_ds, geometry=gpd.points_from_xy(df_ds['lon'], df_ds['lat'], crs='EPSG:4326')) 
    gdf.drop(columns=['lat', 'lon'], inplace=True)
       
    return gdf

    
def get_val_geodataframe(ds, attrs, lc_mask, cl_mask, lat, lon, vectores):
    
    """
    Goals: get the database and remove unnecessary information for validation process
    Parameters
    ----------
        ds: data with error/sunglint pixels changed to nan values
        attrs : metadata information
        lc_mask, cl_mask: mask of landmask and cloundmask 
        lat, lon: latitude and longitude values
        vectores: source list of vectors related to the database 
    Returns-> 
    filtered dataset by supervised vectorization for validation process
    """   
    n_bands = [f'{i:.03f}' for i in attrs['List_Cw_Vnir'].tolist() + attrs['List_Cw_Swir'].tolist()]
    names_bands =  n_bands + ['lat', 'lon', 'cloud_mask', 'land_mask', 'nodata_mask']
    lat = np.expand_dims(lat, axis=1)
    lon = np.expand_dims(lon, axis=1)
    cl_mask = np.expand_dims(cl_mask, axis=1)
    lc_mask = np.expand_dims(lc_mask, axis=1)
    nodata_mask = np.expand_dims((ds.sum(axis=1)>0).astype(int), axis=1)
    
    ds_bands = np.hstack((ds, lat, lon, cl_mask, lc_mask, nodata_mask))
    ds_bands = np.transpose(ds_bands, axes=[2,0,1])
    
    df_ds = pd.DataFrame(ds_bands.reshape(-1, ds_bands.shape[2]), columns=names_bands)  
    
    gdf = gpd.GeoDataFrame(df_ds, geometry=gpd.points_from_xy(df_ds['lon'], df_ds['lat'], crs='EPSG:4326')) 
    gdf.drop(columns=['lat', 'lon'], inplace=True)
     
    supervised_class = gpd.read_file(vectores) 
    gdf_class = gdf.sjoin(supervised_class, how='left') #, how='left'
    gdf_class.loc[gdf_class['nodata_mask']==0, 'label'] = np.nan
    gdf_class.drop(columns=['index_right', 'nodata_mask'], inplace=True)
    return gdf_class


      
def array_totiff(ds, lat, lon, out_mask, bandnames=None):
    
    """
	Goals: write an array to GTiff, from metadata info.
	Parameters
	----------
		ds: numpy array in npy format
		lat, lon: latitude and longitude values 
		vectores: source list of vectors related to the database 
	Returns-> 
	cloud mask in Gtiff
	"""   	
    gcps = []
    # ~ breakpoint()
    if len(ds.shape) == 2:
        rows, cols = ds.shape
        count = 1
    else:
        rows, count, cols = ds.shape
        
    for i in range(0, rows, 10):
        for j in range(0, cols, 10):
            gcps.append(GCP(i, j, lon[i, j], lat[i, j]))
    
    transform = from_gcps(gcps)  
    
    meta = {'driver': 'GTiff',
            'dtype': 'float32',
            'nodata': -9999,
            'width': cols,
            'height': rows,
            'count': count,
            'crs': 4326,
            'transform': transform}
    
        
    with rasterio.open(out_mask, "w", **meta) as dest:
        if count == 1:
            dest.write(ds, 1)
            if bandnames is not None:
                dest.set_band_description(1, bandnames)
        else:
            for i in range(count):
                dest.write(ds[:, i, :], i+1)
                if bandnames is not None and len(bandnames) == count:
                    dest.set_band_description(i+1, bandnames[i])
  





