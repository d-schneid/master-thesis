import json

import numpy as np
import h5py
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)


if __name__ == '__main__':

	file_path = "/Users/i741961/Documents/SAP/Masterthesis/Code/master-thesis/data_preprocessing/data/huawei/code_text/train/samples_0.h5"

	with open('/Users/i741961/Documents/SAP/Masterthesis/Code/master-thesis/data_preprocessing/data/huawei/code_text/train/num_samples_0.json', 'r') as f:
		data = json.load(f)
	num_samples = data['num_samples']

	code_token_lens = []
	text_token_lens = []
	dfg_node_mask_lens = []
	lr_paths_len_lens = []

	with h5py.File(file_path, "r") as h5_file:
		for sample_id in range(1, num_samples):
			sample_group = h5_file[f"sample_{sample_id}"]

			sample_data = {}
			for key in sample_group.keys():
				sample_data[key] = sample_group[key][()]

			for key, value in sample_data.items():
				if key in ['code_token_ids', 'text_token_ids', 'dfg_node_mask', 'lr_paths_len']:
					if key == 'code_token_ids':
						code_token_lens.append(value.shape[0])
					elif key == 'text_token_ids':
						text_token_lens.append(value.shape[0])
					elif key == 'dfg_node_mask':
						dfg_node_mask_lens.append(value.shape[0])
					elif key == 'lr_paths_len':
						lr_paths_len_lens.append(value.shape[0])

	print(f"Max code tokens: {max(code_token_lens)}, Avg: {sum(code_token_lens) / len(code_token_lens):.2f}")
	print(f"Max text tokens: {max(text_token_lens)}, Avg: {sum(text_token_lens) / len(text_token_lens):.2f}")
	print(f"Max DFG nodes: {max(dfg_node_mask_lens)}, Avg: {sum(dfg_node_mask_lens) / len(dfg_node_mask_lens):.2f}")
	print(f"Max LR paths: {max(lr_paths_len_lens)}, Avg: {sum(lr_paths_len_lens) / len(lr_paths_len_lens):.2f}")

	plt.figure(figsize=(12, 8))
	plt.hist(code_token_lens, bins=50, alpha=0.5, label='code_token_ids')
	plt.hist(text_token_lens, bins=50, alpha=0.5, label='text_token_ids')
	plt.hist(dfg_node_mask_lens, bins=50, alpha=0.5, label='dfg_node_mask')
	plt.hist(lr_paths_len_lens, bins=50, alpha=0.5, label='lr_paths_len')
	plt.title("Token Length Distributions")
	plt.xlabel("Sequence Length")
	plt.ylabel("Frequency")
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.show()

