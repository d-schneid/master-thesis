import json
import os

import h5py

from data_preprocessing.datasets.cornstack import CornStack
from data_preprocessing.tasks.pretraining import Pretraining

if __name__ == "__main__":
	task = Pretraining()
	dataset = CornStack(task=task, split='validation')

	input_path = dataset.h5_path_1
	output_path_2 = dataset.h5_path_2
	output_path_3 = dataset.h5_path_3
	output_path_4 = dataset.h5_path_4

	with open(dataset.num_samples_path_1, 'r') as f:
		num_samples = json.load(f)['num_samples']

	third = num_samples // 3
	two_third = 2 * third

	with (h5py.File(input_path, 'r') as input_file,
		  h5py.File(output_path_2, 'w') as out_2,
		  h5py.File(output_path_3, 'w') as out_3,
		  h5py.File(output_path_4, 'w') as out_4):

		for num_sample in range(num_samples):
			if num_sample % 10_000 == 0:
				print(num_sample)
			sample_id = f'sample_{num_sample}'
			sample_group = input_file[sample_id]

			if num_sample < third:
				target_file = out_2
				new_sample_id = f'sample_{num_sample}'
			elif num_sample < two_third:
				target_file = out_3
				new_sample_id = f'sample_{num_sample - third}'
			else:
				target_file = out_4
				new_sample_id = f'sample_{num_sample - two_third}'

			new_group = target_file.require_group(new_sample_id)
			for key, data in sample_group.items():
				new_group.create_dataset(name=key, data=data[()], compression="lzf")

	# Save sample count metadata
	with open(dataset.num_samples_path_2, 'w') as f2:
		json.dump({'num_samples': third}, f2)

	with open(dataset.num_samples_path_3, 'w') as f3:
		json.dump({'num_samples': third}, f3)

	with open(dataset.num_samples_path_4, 'w') as f4:
		json.dump({'num_samples': num_samples - two_third}, f4)
