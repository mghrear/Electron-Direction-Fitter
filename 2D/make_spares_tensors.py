# This script is used to read the processed pickle files from "process_raw_data.py" and perform the following:
# 1. Voxelize the data
# 2. store the data as pytorch sparse tensors, to be analyzed by by a convolutional neural network

import pandas as pd
import numpy as np
import torch


# The data is stored in 100 pickle files each containing 10k electron recoil simulations
files_e = ['~/data/e_dir_fit/3D_processed_data/processed_recoils_'+str(i)+'.pk' for i in range(100) ]


# Here a define the pixel grid parameters
# x/y/z length being kept in cm
eff_l = 2.5
# Voxel size in cm
vox_l = 0.1
# Number of voxels along 1 dim
Npix = round(eff_l*2/vox_l) 
# Tensor dimensions, there is an extra dimension for color which is not used
dim = (Npix,Npix,Npix,1)


# Loop through files
ind = 0 
for file in files_e:

	# Read root file
	df = pd.read_pickle(file)

	# Lists to store sparse tensors and corresponding labels
	labels = []
	sparse_tensors = []

	for index, row in df.iterrows():

		# If recoil escapes fiducial area, skip it
		if np.max(row['x']) >= eff_l or np.min(row['x']) < -eff_l or np.max(row['y']) >= eff_l or np.min(row['y']) < -eff_l or np.max(row['z']) >= eff_l or np.min(row['z']) < -eff_l:
			continue

		# Initialize empty dense tensor
		voxelgrid = np.zeros(dim).astype('uint8')

		# Loop the x, y, z positions in the recoil and fill in the dense tensor
		for x,y,z in zip(row['x'],row['y'],row['z']):
			voxelgrid[int((x+eff_l)/vox_l)][int((y+eff_l)/vox_l)][int((z+eff_l)/vox_l)][0] += 1

		# Convert to pytorch tensor
		voxelgrid = torch.tensor(voxelgrid)
		# Convert to sparse pytorch tensor
		vg = voxelgrid.to_sparse()

		# Store sparse tensor and corresponding label
		sparse_tensors += [vg]
		labels += [row['dir']]

	# Save sparse tensors and labels
	torch.save( sparse_tensors, './sparse_data/sparse_recoils_'+str(ind)+'.pt')
	np.savetxt('./sparse_data/labels_'+str(ind)+'.pt',labels)

	print("Progress: ", ind, '/99')

	ind += 1
