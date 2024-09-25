# Note:
# _**Be careful, the paths in the configuration files must be changed according to your path files structure**_


# Sample config files

This directory contains examples of configuration files in json format 
used for [train](#config_getmodeljson) the model, [validate](#config_getvaljson) 
and [classify](#config_getclassicationjson) new images. 

The way how to call them is detailed in the documentation of each main 
script. Here, an explanation of the content is given.


## config_getmodel.json

This is an example for the configuration file used to generate a new model. 
It is called by the `get_model.py` main script. The structure for this \
file is described below:

```
{
    "model": "knn", 
    "db_folder": "../data/database/",
    "csv_fn": "prisma_database_21_bands.csv",
    "output_model_folder": "../trained_models/",
    "balance": true
}
```
### Mandatory parameters:

**model**: model to be used, options are: knn, rf (Random Forest) and xgboost

**db_folder**: path to the stored database

**csv_fn**: database filename in csv format

**output_model_folder**: path where the model in joblib format will be stored

### optional parameters:

**balance**: boolean parameter. If true, the model will be trained with the 
             same number of pixels for cloud class and no cloud class.



## config_getval.json

This is an example for the configuration file used to validate a model. 
It is called by the `get_validation.py` main script. The structure for this \
file is described below:

```
{
    "img_folder": "../hdf/",
    "val_polygons_folder": "../gpkg/",
    "trained_models_folder": "../trained_models/",
    "output_mask_folder": "../results/mask/knn/",
    "output_log": "../results/log_files/metrics/",
    "model": "knn",
    "stack_tif": true,
    "cloud_prisma": true,
    "npy_matrix": false,
    "prisma_test": false,
    "bands": {
               "VNIR": [480, 559, 650, 660, 742, 762, 840, 942],
               "SWIR": [1114, 1131, 1250, 1386, 1548, 1558, 1651, 1749, 1759, 2072, 2082, 2193, 2203]
              }
}

```

### Mandatory parameters:

**img_folder**: path to the images to use for the validation in hdf5 format.

**val_polygons_folder**: path to the file in gpkg format with the polygons
                         for cloud and no cloud classes for each image used 
                         for the validation.

**trained_models_folder**: path to the trained model in joblib format

**output_mask_folder**: path to the output directory

**output_log**: path to the output log file which will contain the values 
                for the different accuracy metrics estimated for the model

**model**: model to be used, options are: knn, rf (Random Forest) and xgboost

**bands**:  dictionary with list of the bands for each spectral region to
            be used for the classification. They must be compliant with the \
            &emsp;&emsp;&emsp;&nbsp; bands used to train the model. If 
            the model has been trained with all available bands, this 
            parameter can be optional.

### optional parameters:

**stack_tif**: boolean parameter. If true, original bands are going to be
               stacked to the output mask.

**cloud_prisma**: boolean parameter. If true, original prisma cloud mask 
                  is going to be stacked to the output mask.

**npy_matrix**: boolean parameter. If true, a numpy array is also generated 
                in the output folder

**prisma_test**: boolean parameter. If true, the metrics will be computed 
                 also for the original prisma cloud mask.

**bands**: dictionary with list of the bands for each spectral region to
           be used for the classification. They must be compliant with the \
           &emsp;&emsp;&emsp;&nbsp; bands used to train the model. If the 
           model has been trained with a subset of bands, this parameter 
           becomes mandatory.

## config_getclassification.json

This is an example for the configuration file used to classify a new image. 
It is called by the `get_classification.py` main script. The structure for \
this file is described below:

```
{
    "img_folder": "../hdf/",
    "trained_models_folder": "../trained_models/",
    "output_mask_folder": "../results/",
    "model": "knn",
    "stack_tif": true,
    "cloud_prisma": true,
    "npy_matrix": false,
    "bands": {
               "VNIR": [480, 559, 650, 660, 742, 762, 840, 942],
               "SWIR": [1114, 1131, 1250, 1386, 1548, 1558, 1651, 1749, 1759, 2072, 2082, 2193, 2203]
              }
}

```

### Mandatory parameters:

**img_folder**: path to the images to classify in hdf5 format.

**trained_models_folder**: path to the trained model in joblib format

**output_mask_folder**: path to the output directory

**model**: model to be used, options are: knn, rf (Random Forest) and xgboost

**bands**:  dictionary with list of the bands for each spectral region to
            be used for the classification. They must be compliant with the \
            &emsp;&emsp;&emsp;&nbsp; bands used to train the model. If 
            the model has been trained with all available bands, this 
            parameter can be optional.

### optional parameters:

**stack_tif**: boolean parameter. If true, original bands are going to be
               stacked to the output mask.

**cloud_prisma**: boolean parameter. If true, original prisma cloud mask 
                  is going to be stacked to the output mask.

**npy_matrix**: boolean parameter. If true, a numpy array is also generated 
                in the output folder

**bands**: dictionary with list of the bands for each spectral region to
           be used for the classification. They must be compliant with the \
           &emsp;&emsp;&emsp;&nbsp; bands used to train the model. If the 
           model has been trained with a subset of bands, this parameter 
           becomes mandatory.

