# Electron-Direction-Fitter
A 3D convolutional neural network for determining the initial direction of a electron recoil.

## explore_raw_data.ipynb:
Exploratory jupyter notebook which demonstrates the following:
1. Reading raw simulation finles
2. Visualizing the raw simulation files 
3. Obtaining an random direction drawn from a isotropic distribution
4. Rotating tracks to the random direction and applying diffusion
5. Mean-centering the tracks 

## process_raw_data.py:
Python implementation of 'explore_raw_data.ipynb' to process all raw degrad simulations.

## explore_processed_data.ipynb:
This script is used to read the processed simulation as pickle files and perform the following:
1. Demonstrate that the simulations are isotropic
2. Demostrate that the labels (true directions) are correct
3. Demonstrate how the simulations are further processed into grids that can be analyzed by a 3D CNN
4. Visualize the fully processed simulations along with the true direction
5. Demonstrate how simulations can be saved as sparse tensors which take up less space
6. Demstrates how the spare tensors can be load
7. Demonstrates how a data generator can be used to convert sparse tensors into dense tensors in a batch-by-batchn basis (so that they do not take up too much space)

## makes_sparse_tensors.py:
Python implementation of 'explore_processed_data.ipynb' to convert all processed simulations into sparse tensors that can be stored efficiently and loaded into a NN as dense tensors in a batch-by-batch bases.
