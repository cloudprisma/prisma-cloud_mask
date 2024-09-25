# PRISMA - Cloud_mask

# 🛰️ SAPP4VU: Sviluppo di Algoritmi prototipali Prisma per la Stima del Danno Ambientale e della Vulnerabilità alla Land Degradation

## 🚀 "Implementazione di algoritmi numerico-statistici per la caratterizzazione e rimozione del rumore e per la cloud detection in immagini iperspettrali.”  - Part 2

## Description

This repository contains the Python sources of the Prisma basic processing and some parts were adapted from the original code developed by [[1]](Vanhellemonthttps://www.sciencedirect.com/science/article/pii/S0034425718303481) - see the [ACOLITE: generic atmospheric correction module - for PRISMA](https://github.com/acolite/acolite) 

# How it works?

![process](https://github.com/user-attachments/assets/3e311ac7-de17-4f58-af6d-14e2d3140abe)


Project Structure:
In this project you will find:

requirements.txt it contains all the necessary libraries;


![_Diagrama de flujo - prisma_program](https://github.com/user-attachments/assets/28528974-bada-4f87-ab44-9685b03012a5)
- scripts contains a modular code;
- trained_models contains the best model based on the dice score and the last trained model;
- output_samples contains some predictions and the corresponding ground truth images.

Prepare environment

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
      $ pip install pyvsnr
  ```
  or  If you use the requirements.txt file:
  ```
      $ pip install -r requirements.txt
  ```

  4. Download the scripts available here ( _*main.py_ and _*functions_he5.py_ ) and save them into the same directory.
  5. Then, once at the directory, execute the next command in a terminal e.g.
 ```
      $ python main.py -if ./PRS_L1_STD_OFFL_20210922101425_20210922101429_0001.he5 -s HRC -sr VNIR -nt additive 
  ```
# git clone source
!git clone https://github.com/cloudprisma/prisma-cloud_mask

# Results

![image](https://github.com/user-attachments/assets/d4aa27a3-53fd-480e-9022-9c6393ed1c5d)

# Results


![image](https://github.com/user-attachments/assets/2151f774-59c5-4e5c-8d17-c2a33dbd5950)

# 📝 Authors information
This is adapted to Python from the original Matlab codes developed by:
 - [x] see the [ACOLITE: generic atmospheric correction module - for PRISMA](https://github.com/acolite/acolite) 

 - [x] José Bioucas-Dias and José Nascimento

All credit goes to the original author.

In case you use the results of this code with your article, please don't forget to cite:

- [x] Fehrenbach, Jérôme, Pierre Weiss, and Corinne Lorenzo. "Variational algorithms to remove stationary noise: applications to microscopy imaging." IEEE Transactions on Image Processing 21.10 (2012): 4420-4430.
- [x] Fehrenbach, Jérôme, and Pierre Weiss. "Processing stationary noise: model and parameter selection in variational methods." SIAM Journal on Imaging Sciences 7.2 (2014): 613-640.
- [x] Escande, Paul, Pierre Weiss, and Wenxing Zhang. "A variational model for multiplicative structured noise removal." Journal of Mathematical Imaging and Vision 57.1 (2017): 43-55.
- [x] Bioucas-Dias, J. and  Nascimento, J.  "Hyperspectral subspace identification", IEEE Transactions on G
