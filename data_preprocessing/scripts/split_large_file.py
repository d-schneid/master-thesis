import json

import h5py

from data_preprocessing.datasets.huawei import Huawei
from data_preprocessing.tasks.code_text import CodeText

if __name__ == "__main__":
	task = CodeText()
	dataset = Huawei(task=task, split='validation')

	input_path = dataset.h5_path
	output_path_0 = dataset.h5_path_0
	output_path_1 = dataset.h5_path_1

	with open(dataset.num_samples_path, 'r') as f:
		num_samples = json.load(f)['num_samples']

	half = num_samples // 2

	with (h5py.File(input_path, 'r') as input_file,
		  h5py.File(output_path_0, 'w') as out_0,
		  h5py.File(output_path_1, 'w') as out_1):

		for num_sample in range(num_samples):
			if num_sample % 10_000 == 0:
				print(num_sample)
			sample_id = f'sample_{num_sample}'
			sample_group = input_file[sample_id]

			if num_sample < half:
				target_file = out_0
				new_sample_id = f'sample_{num_sample}'
			else:
				target_file = out_1
				new_sample_id = f'sample_{num_sample - half}'

			new_group = target_file.require_group(new_sample_id)
			for key, data in sample_group.items():
				new_group.create_dataset(name=key, data=data[()], compression="lzf")

	# Save sample count metadata
	with open(dataset.num_samples_path_0, 'w') as f0:
		json.dump({'num_samples': half}, f0)

	with open(dataset.num_samples_path_1, 'w') as f1:
		json.dump({'num_samples': num_samples - half}, f1)
