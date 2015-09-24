import pandas as pd
import numpy as np
import json
import csv
import glob


def compile(glob_pattern, save_name):
	files = glob.glob(glob_pattern)
	results = []


	for json_file in CSC_files:
		with open(json_file, 'r') as f:
			loaded_eccentricities = json.load(f)['results']
			for eccentricity in loaded_eccentricities:
				for area in eccentricity['areas']:
					CSC_results.append([
							eccentricity['eccentricity'], #eccentricity\ 
							np.around(area['area'],decimals=2),#area of the sample 
							area['N'], #number of cells in the subsample\ 
							area['subsample_100']['E_of_pi'], #E_of_pi of the subsample
							area['subsample_100']['S'], # S of the subsample\ 
							area['subsample_100']['d'] # d of the subsample
						])
				#endfor
			#endfor
		#endwith
	#endfor
	df = pd.DataFrame(CSC_results, columns=['eccentricity', 'area', 'N', 'E_of_pi', 'S', 'd'])
	unique_eccentricities = pd.unique(df.eccentricity.ravel())
	unique_areas = pd.unique(df.area.ravel())


	averaged_dataset = []

	for ecc in unique_eccentricities:
		subset = df[df['eccentricity'] == ecc]
		for area in unique_areas:
			area_subset = subset[subset['area'] == area].mean().values
			averaged_dataset.append(area_subset)
		#endfor
	#enedfor
	
	return pd.DataFrame(averaged_dataset, columns=['eccentricity', 'area', 'N', 'E_of_pi', 'S', 'd'])