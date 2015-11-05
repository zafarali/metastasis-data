import numpy as np
import sys
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# custom color map that goes from Blue --> Red

cmap = [(255,0,2),
(228,0,28),
(201,1,54),
(170,86,82),
(138,199,110),
(108,199,138),
(79,85,168),
(51,0,197),
(25,0,227),
(0,0,255)]

color = mpl.colors.ListedColormap(np.array(cmap)/float(255)).colors
# mpl.rc('text', usetex = True)


def H(n):
	"""Returns an approximate value of n-th harmonic number.

	   http://en.wikipedia.org/wiki/Harmonic_number
	"""
	# Euler-Mascheroni constant
	gamma = 0.57721566490153286060651209008240243104215933593992
	return gamma + np.log(n) + 0.5/n - 1./(12*n**2) + 1./(120*n**4)

# vectorize this to use with numpy arrays
H_vectorized = np.vectorize(H)

if len(sys.argv) < 4:
	print 'ERROR MUST HAVE:\n'
	print 'python graph_pipeline.py file_path plot_type title out_folder\n'
	print 'file_path = folder which contains the csv'
	print 'plot_type = S / pi / SH '
	print 'out_folder = folder to save files in'
	print 'OPTIONAL: selected_eccs_only = FALSE / TRUE'
	sys.exit(-1)

file_path = sys.argv[1] # path to the files / colelction of files
plot_type = sys.argv[2] # S or pi
# title = sys.argv[3] # title of these plots
out_folder = sys.argv[3]




if len(sys.argv) == 5 and sys.argv[4] == 'TRUE':
	selected_eccs = [0.1, 0.9]
else:
	selected_eccs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


MODELTYPES = ['1stOrder', 'MutationResponse', 'CSC', 'MDCSC']
TITLES = ['Single Driver Model', 'Multiple Driver Model', 'Cancer Stem Cell Model', 'Multiple Dirver Cancer Stem Cell Model']
RATIOS = ['base', 'cancer_80', 'cancer_90', 'cancer_only']
RATIO_TITLES = ['0%', '80%', '90%', '100%']

eccentricities_to_compare = {}
# eccentricities_to_compare['areas'] = pd.unique(df['area'])

eccentricities_to_compare['1stOrder'] = {}
eccentricities_to_compare['CSC'] = {}
eccentricities_to_compare['MDCSC'] = {}
eccentricities_to_compare['MutationResponse'] = {}

# loop over all models
for model, title in zip(MODELTYPES, TITLES):
	# loop over all ratios
	for ratio,ratio_title in zip(RATIOS, RATIO_TITLES):
		plt.clf()
		plt.figure()
		# load the data
		df = pd.read_csv(file_path+model+'_'+ratio+'.csv', index_col=0)

		# extract unique eccentricies
		unique_eccentricities = pd.unique(df['eccentricity'])

		# do the plotting
		for index,ecc in enumerate(unique_eccentricities):

			if ecc not in selected_eccs:
				continue

			subset = df[df['eccentricity']==ecc]
			x = subset['area']

			if plot_type == 'SH':
				y = subset['S'] / H_vectorized(subset['N'])
				ratio_save = subset['S'].values / H_vectorized(subset['N'])
				this_ylabel = 'S/H'

			elif plot_type == 'pi':
				y = subset['E_of_pi'] 
				ratio_save = subset['E_of_pi'].values
				this_ylabel = 'pi'

			elif plot_type == 'S':
				y = subset['S'] 
				ratio_save = subset['S'].values
				this_ylabel = 'S'
			elif plot_type == 'D':
				y = subset['E_of_pi'] - (subset['S'] / H_vectorized(subset['N']))
				ratio_save = subset['E_of_pi'].values - (subset['S'].values / H_vectorized(subset['N']))
				this_ylabel = 'D'


			ecc = str(np.around(ecc, decimals=1))
			eccentricities_to_compare[model][ecc] = eccentricities_to_compare[model].get(ecc, {})
			eccentricities_to_compare[model][ecc][ratio_title] = ratio_save
			plt.plot(x,y, 'x-', label='e='+str(ecc), color=color[index])

		plt.legend(loc='upper center', bbox_to_anchor=(1.15, 1), fancybox=True, ncol=1)
		plt.title(this_ylabel+' vs. Area of Sample, ' + title + '(t = '+ratio_title+')')
		plt.ylabel(this_ylabel)
		plt.xlabel('Area of Sample')

		save_fig = out_folder+model+'_'+ratio+'_'+plot_type
		# plt.savefig( save_fig+'_dpi100.png' , format='png', transparent=False, dpi=100, bbox_inches='tight')
		# plt.savefig( save_fig+'_dpi100.pdf' , format='pdf', transparent=False, dpi=100, bbox_inches='tight')
		# plt.savefig( save_fig+'_dpi100.svg' , format='svg', transparent=False, dpi=100, bbox_inches='tight')
		plt.savefig( save_fig+'_dpi200.png' , format='png', transparent=False, dpi=200, bbox_inches='tight')
		# plt.savefig( save_fig+'_dpi200.pdf' , format='pdf', transparent=False, dpi=200, bbox_inches='tight')
		# plt.savefig( save_fig+'_dpi200.svg' , format='svg', transparent=False, dpi=200, bbox_inches='tight')
		# plt.savefig( save_fig+'_dpi300.png' , format='png', transparent=False, dpi=300, bbox_inches='tight')
		# plt.savefig( save_fig+'_dpi300.pdf' , format='pdf', transparent=False, dpi=300, bbox_inches='tight')
		# plt.savefig( save_fig+'_dpi300.svg' , format='svg', transparent=False, dpi=300, bbox_inches='tight')
		print 'Saved '+save_fig


	x = pd.unique(df['area'])
	for ecc, obj in sorted(eccentricities_to_compare[model].items(), key=lambda x: x[0]):
		plt.figure()
		plt.plot(x, obj['0%'], label='0%', color='y', lw=2)
		plt.plot(x, obj['80%'], label='80%', color='g', lw=2)
		plt.plot(x, obj['90%'], label='90%', color='b', lw=2)
		plt.plot(x, obj['100%'], label='100%', color='r', lw=2)
		plt.legend(loc='lower right')

		plt.title('Comparing ' + this_ylabel + ', ' + title + ' (e='+str(ecc)+')')
		plt.xlabel('Area of Sample')
		plt.ylabel(this_ylabel)
		save_fig = out_folder+model+'_ecc_compare_'+str(ecc)

		# plt.savefig( save_fig+'_dpi100.png' , format='png', transparent=False, dpi=100, bbox_inches='tight')
		# plt.savefig( save_fig+'_dpi100.pdf' , format='pdf', transparent=False, dpi=100, bbox_inches='tight')
		# plt.savefig( save_fig+'_dpi100.svg' , format='svg', transparent=False, dpi=100, bbox_inches='tight')
		plt.savefig( save_fig+'_dpi200.png' , format='png', transparent=False, dpi=200, bbox_inches='tight')
		# plt.savefig( save_fig+'_dpi200.pdf' , format='pdf', transparent=False, dpi=200, bbox_inches='tight')
		# plt.savefig( save_fig+'_dpi200.svg' , format='svg', transparent=False, dpi=200, bbox_inches='tight')
		# plt.savefig( save_fig+'_dpi300.png' , format='png', transparent=False, dpi=300, bbox_inches='tight')
		# plt.savefig( save_fig+'_dpi300.pdf' , format='pdf', transparent=False, dpi=300, bbox_inches='tight')
		# plt.savefig( save_fig+'_dpi300.svg' , format='svg', transparent=False, dpi=300, bbox_inches='tight')
		print 'Saved '+save_fig


formatter = {}
for model, details in eccentricities_to_compare.items():
	if model == 'areas':
		continue
	for ecc, data in details.items():
		formatter[ecc] = formatter.get(ecc, {})
		formatter[ecc][model] = data

for ecc, data in formatter.items():
	for ratio in RATIO_TITLES:
		plt.figure()
		plt.plot(x, data['1stOrder'][ratio], 'r', label='SD', lw=2)
		plt.plot(x, data['CSC'][ratio], 'b', label='CSC', lw=2)
		plt.plot(x, data['MutationResponse'][ratio], 'g', label='MD', lw=2)
		plt.plot(x, data['MDCSC'][ratio], 'g', label='MDCSC', lw=2)
		plt.title('Comparison of ' + this_ylabel + ' between models (e=' + ecc + ', t='+ratio+')')
		plt.legend(loc='lower right')
		plt.ylabel(this_ylabel)
		plt.xlabel('Area of Sample')
		save_fig = out_folder+'ecc_compare_'+str(ecc)+'_multimodel_ratio'+ratio

		# plt.savefig( save_fig+'_dpi100.png' , format='png', transparent=False, dpi=100, bbox_inches='tight')
		# plt.savefig( save_fig+'_dpi100.pdf' , format='pdf', transparent=False, dpi=100, bbox_inches='tight')
		# plt.savefig( save_fig+'_dpi100.svg' , format='svg', transparent=False, dpi=100, bbox_inches='tight')
		plt.savefig( save_fig+'_dpi200.png' , format='png', transparent=False, dpi=200, bbox_inches='tight')
		# plt.savefig( save_fig+'_dpi200.pdf' , format='pdf', transparent=False, dpi=200, bbox_inches='tight')
		# plt.savefig( save_fig+'_dpi200.svg' , format='svg', transparent=False, dpi=200, bbox_inches='tight')
		# plt.savefig( save_fig+'_dpi300.png' , format='png', transparent=False, dpi=300, bbox_inches='tight')
		# plt.savefig( save_fig+'_dpi300.pdf' , format='pdf', transparent=False, dpi=300, bbox_inches='tight')
		# plt.savefig( save_fig+'_dpi300.svg' , format='svg', transparent=False, dpi=300, bbox_inches='tight')
		print 'Saved '+save_fig

#     plt.plot(x, data['MutationResponse']['0%'])

