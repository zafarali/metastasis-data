import pandas as pd
import numpy as np
import json
import csv
import glob

default_patterns = [
	( './CSC/*/pipe*/sample_1*/*.json', 'CSC_base.csv'), 
	( './CSC/*/pipe*/sample_2*/*.json', 'CSC_cancer_only.csv'),
	( './CSC/*/pipe*/sample_3*/*.json', 'CSC_cancer_80.csv'),
	( './CSC/*/pipe*/sample_4*/*.json', 'CSC_cancer_90.csv'),
	( './MDCSC/*/pipe*/sample_1*/*.json', 'MDCSC_base.csv'), 
	( './MDCSC/*/pipe*/sample_2*/*.json', 'MDCSC_cancer_only.csv'),
	( './MDCSC/*/pipe*/sample_3*/*.json', 'MDCSC_cancer_80.csv'),
	( './MDCSC/*/pipe*/sample_4*/*.json', 'MDCSC_cancer_90.csv'),
	( './1stOrder/*/pipe*/sample_1*/*.json', '1stOrder_base.csv'), 
	( './1stOrder/*/pipe*/sample_2*/*.json', '1stOrder_cancer_only.csv'),
	( './1stOrder/*/pipe*/sample_3*/*.json', '1stOrder_cancer_80.csv'),
	( './1stOrder/*/pipe*/sample_4*/*.json', '1stOrder_cancer_90.csv'),
	( './MutationResponse/*/pipe*/sample_1*/*.json', 'MutationResponse_base.csv'), 
	( './MutationResponse/*/pipe*/sample_2*/*.json', 'MutationResponse_cancer_only.csv'),
	( './MutationResponse/*/pipe*/sample_3*/*.json', 'MutationResponse_cancer_80.csv'),
	( './MutationResponse/*/pipe*/sample_4*/*.json', 'MutationResponse_cancer_90.csv')

]

def compile_from_json(glob_pattern, save_name=None):
	files = glob.glob(glob_pattern)
	results = []


	for json_file in files:
		with open(json_file, 'r') as f:
			loaded_eccentricities = json.load(f)['results']
			for eccentricity in loaded_eccentricities:
				for area in eccentricity['areas']:
					results.append([
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
	df = pd.DataFrame(results, columns=['eccentricity', 'area', 'N', 'E_of_pi', 'S', 'd'])
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

	if save_name:
		pd.DataFrame(averaged_dataset, columns=['eccentricity', 'area', 'N', 'E_of_pi', 'S', 'd']).to_csv(save_name)

	return pd.DataFrame(averaged_dataset, columns=['eccentricity', 'area', 'N', 'E_of_pi', 'S', 'd'])


if __name__ == '__main__':
	for pattern in default_patterns:
		compile_from_json(*pattern)
