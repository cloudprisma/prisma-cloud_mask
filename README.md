# PRISMA - Cloud_mask

# 🛰️ SAPP4VU: Sviluppo di Algoritmi prototipali Prisma per la Stima del Danno Ambientale e della Vulnerabilità alla Land Degradation

## 🚀 "Implementazione di algoritmi numerico-statistici per la caratterizzazione e rimozione del rumore e per la cloud detection in immagini iperspettrali.”  - Part 2

## Description

This repository contains the Python sources of the Prisma basic processing for cloud classification. Some parts for the preprocessing were adapted from the original code developed by [[1]](Vanhellemonthttps://www.sciencedirect.com/science/article/pii/S0034425718303481) - see the [ACOLITE: generic atmospheric correction module - for PRISMA](https://github.com/acolite/acolite) 

# How it works?

![process_cloud](https://github.com/user-attachments/assets/4947ba2b-00e0-4374-88af-9d3f57961fbb)

# Note:
Due to the weight of the PRISMA images, routines for extracting the database were excluded. However, the database is available in the following link: [Database](https://github.com/cloudprisma/prisma-cloud_mask/blob/main/data/database.md).
  
# Prisma program Structure:
In this project you will find:

* requirements.txt it contains all the necessary libraries;

* scripts contains a modular code:
![_Diagrama de flujo - prisma_program](https://github.com/user-attachments/assets/28528974-bada-4f87-ab44-9685b03012a5)
- trained_models: contains the best model based on optuna optimization. 
Additionaly you will find two directories. First one called [configuration_files](https://github.com/cloudprisma/prisma-cloud_mask/tree/main/configuration_files), which provides examples to set the different input files to run the main classification scripts.
Second one, called [sample_validation-data](https://github.com/cloudprisma/prisma-cloud_mask/tree/main/sample_validation-data) provides the access to sample prisma dataset and its vector masks.

# Prepare environment

  1. Create an environment, for instance:
  ```
    $ pip install virtualenv
    $ python -m venv <virtual-environment-name>
  ```
  
  2. Activate your virtual environment:
  ```
      $ source env/bin/activate
  ```
  3.  Install the requirements in the Virtual Environment, you can easily just pip install the libraries. For example:
  ```
      $ pip install numpy
  ```
  or  If you use the requirements.txt file:
  ```
      $ pip install -r requirements.txt
  ```

  4. Download the scripts available here and save them into the same directory or try via git clone source:
  ```
      $ git clone https://github.com/cloudprisma/prisma-cloud_mask
  ```
  # Note:
  Given its weight, some files are attached as google drive link. Do not forget to download them:

  - [Database](https://github.com/cloudprisma/prisma-cloud_mask/blob/main/data/database.md)
  - [sample_validation_data/gpkg/](https://github.com/cloudprisma/prisma-cloud_mask/blob/main/sample_validation-data/gpkg/gpkg_files.md) folder
  - [sample_validation_data/hdf/](https://github.com/cloudprisma/prisma-cloud_mask/blob/main/sample_validation-data/hdf/prisma_hdf_files.md) folder
  - [trained_models/knn.joblib](https://github.com/cloudprisma/prisma-cloud_mask/blob/main/trained_models/knn_joblib.md)

# Run the scripts
  Locate at the main directory. Once there, execute the next command in a terminal, for example, to run
  the classification script ([get_classification.py](https://github.com/cloudprisma/prisma-cloud_mask/blob/main/main/get_classification.py)), you can run the following line:
  ```
      $ python get_classification.py -i <Path to the config_getclassification.json file>
  ```
  Additionally, you can access to the help of each script for generate a model ([get_model.py](https://github.com/cloudprisma/prisma-cloud_mask/blob/main/main/get_model.py)), validation ([get_validation.py](https://github.com/cloudprisma/prisma-cloud_mask/blob/main/main/get_validation.py)) or classification ([get_classification.py](https://github.com/cloudprisma/prisma-cloud_mask/blob/main/main/get_classification.py)) using the `-h` tag:
 
 ```
      $ python get_model.py -h
 ```
 

# Sample Results
Below, some sample results for Validation and Classification are shown by using the provided [models](https://github.com/cloudprisma/prisma-cloud_mask/tree/main/trained_models) which were trained with the following 21 spectral bands: 

```
"bands": {
          "VNIR": [480, 559, 650, 660, 742, 762, 840, 942],
          "SWIR": [1114, 1131, 1250, 1386, 1548, 1558, 1651, 1749, 1759, 2072, 2082, 2193, 2203]
         }
```
## Validation

![image](https://github.com/user-attachments/assets/d4aa27a3-53fd-480e-9022-9c6393ed1c5d)

## Classification 

![image](https://github.com/user-attachments/assets/2151f774-59c5-4e5c-8d17-c2a33dbd5950)

# 📝 Authors information
This repository contains the Python sources of the Prisma basic processing and some parts were adapted from the original code developed by:
 - [x] see the [ACOLITE: generic atmospheric correction module - for PRISMA](https://github.com/acolite/acolite) 

All credit goes to the corresponding author.
