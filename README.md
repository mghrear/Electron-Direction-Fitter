# Electron-Direction-Fitter
A convolutional neural network for determining the initial direction of a electron recoil

## 2D
The 2D folder uses projected tracks and a 2DCNN implemented via keras

###explore_raw_data.ipynb:
Exploratory jupyter notebook which demonstrates the following:
1. Reading raw simulation finals 
2. Visualizing the raw simulation files 
3. Obtaining an random direction drawn from a isotropic distribution
4. Rotating tracks to the random direction and applying diffusion
5. Mean-centering the tracks  




## 3D
The 2D folder uses projected tracks and a 2DCNN implemented via keras

###explore_raw_data.ipynb:
Exploratory jupyter notebook which demonstrates the following:
1. Reading raw simulation finals 
2. Visualizing the raw simulation files 
3. Obtaining an random direction drawn from a isotropic distribution
4. Rotating tracks to the random direction and applying diffusion
5. Mean-centering the tracks 

###explore_raw_data.ipynb:
This script is used to read the degrad simulation as root files and perform the following:
1. Isotropize the simulations and creat a label for the true recoil direction
2. Diffuse the simulations
3. mean-center the simulations
4. Store the simulations as pickles 