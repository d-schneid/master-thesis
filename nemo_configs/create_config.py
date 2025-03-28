import copy
from omegaconf import OmegaConf, MISSING


def create_model_config():
	common_config = OmegaConf.create({
		'vocab_size': MISSING,
		'block_size': MISSING,
		'n_layer': MISSING,
		'n_embd': MISSING,
		'n_head': MISSING,
	})
	model_config = OmegaConf.create({
		'models': common_config
	})

	temp_config = copy.deepcopy(model_config)
	temp_config.model.vocab_size = 10
	temp_config.model.block_size = 4
	temp_config.model.n_layer = 1
	temp_config.model.n_embd = 32
	temp_config.model.n_head = 4

	temp_config = OmegaConf.create(OmegaConf.to_container(temp_config, resolve=True))
	print(OmegaConf.to_yaml(temp_config))
	OmegaConf.save(temp_config, 'model_config.yaml')

	return temp_config


def create_data_config(cfg):
	OmegaConf.set_struct(cfg.model, False)
	cfg.model.data_path = '../data/pretraining/'
	OmegaConf.set_struct(cfg.model, True)

	train_ds = OmegaConf.create({
		'data_path': '${models.data_path}',
		'block_size': '${models.block_size}',
		'crop': [0, int(1e6)],
		'batch_size': 64,
		'shuffle': True,
	})

	validation_ds = OmegaConf.create({
		'data_path': '${models.data_path}',
		'block_size': '${models.block_size}',
		'crop': [int(1e6), int(50e3)],
		'batch_size': 4,
		'shuffle': False,
	})

	test_ds = OmegaConf.create({
		'data_path': '${models.data_path}',
		'block_size': '${models.block_size}',
		'crop': [int(1.05e6), int(100e3)],
		'batch_size': 4,
		'shuffle': False,
	})

	OmegaConf.set_struct(cfg.model, False)

	cfg.model.train_ds = train_ds
	cfg.model.validation_ds = validation_ds
	cfg.model.test_ds = test_ds

	OmegaConf.set_struct(cfg.model, True)

	temp_config = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
	print(OmegaConf.to_yaml(temp_config))
	OmegaConf.save(temp_config, 'model_config.yaml')

if __name__ == '__main__':
	temp_config = create_model_config()
	create_data_config(temp_config)