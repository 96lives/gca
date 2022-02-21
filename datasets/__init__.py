from datasets.cgca_autoencoder_dataset import (
	AutoencoderShapenetDataset,
)
from datasets.transition_dataset import (
	TransitionShapenetDataset,
)

DATASET = {
	# autoencoder datasets
	AutoencoderShapenetDataset.name: AutoencoderShapenetDataset,

	# transition datasets
	TransitionShapenetDataset.name: TransitionShapenetDataset,
}
