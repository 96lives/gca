# Generative Cellular Automata (GCA) 



![Alt text](media/generations.gif?raw=true "Title")

![Alt text](media/scene_completions.gif?raw=true "Title")

This repository contains code for our work ["Learning to Generate 3D Shapes with Generative Cellular Automata"](https://openreview.net/forum?id=rABUmU3ulQh) and [Probabilistic Implicit Scene Completion](https://openreview.net/forum?id=BnQhMqDfcKG), which are accepted to ICLR 2021 and 2022 (spotlight), respectively. The first paper introduces a model named Generative Cellular Automata (GCA), which formulates the shape generation process as sampling from the transition kernel of a Markov chain, where the sampling chain eventually evolves to the full shape of the learned distribution. The transition kernel employs the local update rules of cellular automata, effectively reducing the search space in a high-resolution 3D grid space by exploiting the connectivity and sparsity of 3D shapes. 

The second paper introduces a model name continuous Generative Cellular Automata (cGCA), which extends GCA to produce continuous geometry from incomplete point cloud. Instead of learning a transition kernel on sparse voxels as in GCA, cGCA learns the transition kernel operating on sparse voxel embedding, which additionally contains a local latent code for each occupied cell. Decoding the last state (sparse voxel embedding) produces continuous surface. Since cGCA extends the scalability, our work is the first work to tackle the problem of completing multiple continuous surfaces in scene level. 

The repository currently contains pretrained models and datasets for the experiments on ShapeNet and ShapeNet scenes in [Probabilistic Implicit Scene Completion](https://openreview.net/forum?id=BnQhMqDfcKG). Please contact 96lives@snu.ac.kr if you have any questions :)




## Installation & Data Preparation

1. **Anaconda environment installations for training & testing**

```
conda create -n gca python=3.8
conda activate gca

# install torch
# for cuda 11.2
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
# for cuda 10.2
pip install torch==1.7.1 torchvision==0.8.2

# install MinkowskiEngine (sparse tensor processing library)
# the model was trained on v0.5.0, but the code runs on v0.5.4 (the latest version as of 22/04/18) as well
conda install openblas-devel -c anaconda 
export CXX=g++-7
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"

# install all other requirements
pip install -r requirements.txt

# install torch-scatter
# for cuda 10.2 (--no-index option is quite crucial), for other cuda versions you can find installation guide in https://github.com/rusty1s/pytorch_scatter
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.1+cu101.html
```
The repo was tested with NVIDIA 2080ti GPU (11GB). 



2. **Data preparation and Pretrained Models**

[TODO: Link] contains the datasets (sdf & preprocessed sparse voxel embedding) and pretrained models. Place the files with directory as below. We call the shapenet scene datasets as synthetic as abbreviation. 

```
\cgca
   - main.py
   ...
   - data/
      - shapenet_sdf/
      	- sofa
      	- chair
      	- table
      - synthetic/ (we refer to shapenet scene as synthetic for abbreviation)
      	- 
      - embeddings/   (contains preprocessed sparse voxel embeddings)
      	- shapenet/
      		- sofa-vox_64-sdf-step_700k/
         	- chair-vox_64-sdf-step_500k/
         	- table-vox_64-sdf-step_500k/
         - synthetic/
         	- vox_64-step_10k/
   - pretrained_models/
      - sofa_transition
         ...
```

For shapenet embeddings we use 



# Training 

We provide training scripts for GCA and cGCA on shapenet/shapenet scene dataset.



## Training GCA

![Alt text](media/gca_method_overview.png?raw=true "Title")

You can train the GCA by running,  

```
python main.py --config configs/gca-sofa.yaml -l log/gca-sofa
```

For other datasets/configurations you may use other configs. Note that GCA only uses the occupancies of the sparse voxels in the dataset, but the released dataset contains the local embeddings as well. If you want to create your own dataset, you only need coordinates of occupied cells of the surface.



## Training cGCA

![Alt text](media/cgca_method_overview.jpg?raw=true "Title")

Training cGCA works in 2 steps. 

1. **Training autoencoder for sparse voxel embedding**

To train the autoencoder model, run
```
python main.py --config configs/cgca_autoencoder-sofa.yaml -l log/autoencoder-sofa
```

2. **Training the cGCA transition model**

You do not need to train the autoencoder from scratch to train the transition model. We have already preprocessed the ground truth sparse voxel embedding for the dataset in the above data preparation step. The sparse voxel embeddings and pretrained autoencoder models (with configs) are in `data/embeddings/shapenet/{obj_class}` for shapenet dataset or `data/embeddings/synthetic` for shapenet scene dataset.

To train the transition model, run

```
python main.py --config configs/cgca_transition-sofa.yaml -l log/transition-sofa
```

3. **Log visualization** 

The log files for the tensorboard visualization is available on the `log` directory.
To view the logs, run

```
tensorboard --logdir ./log
```
and enter the corresponding website with port on your web browser.



# Testing 

Download the pretrained models as described in the above.



## Testing GCA



## Testing cGCA (TODO: make it in one script)



For the efficiency of vRAM usage, we test the cGCA in 2 step procedure.

1. Creating the last sparse voxel embedding 

This procedure iterates by transitions and caches the last sparse voxel embedding ($s^{T+T'}$  in the main paper).
```
python main.py --test --resume-ckpt pretrained_models/sofa_transition/ckpts/ckpt-step-200000 --override "cache_only=True" -l log/sofa_test
```
2. Testing the metrics (this also generates meshes)

This procedure decodes the cached sparse voxel embedding $s^{T+T'}$ .
You can find the reconstructed meshes (obj files) in the `log/sofa_test/test_save/step-200000/mesh/initial_mesh` folder.

```
python main.py --test --resume-ckpt pretrained_models/sofa_transition/ckpts/ckpt-step-200000 --override "cache_only=False" -l log/sofa_test
```





# Citation

If you find this repo useful for your research or use any part of the code, please cite 

```
@inproceedings{
	zhang2021gca,
	title={Learning to Generate 3D Shapes with Generative Cellular Automata},
	author={Dongsu Zhang and Changwoon Choi and Jeonghwan Kim and Young Min Kim},
	booktitle={International Conference on Learning Representations},
	year={2021},
	url={https://openreview.net/forum?id=rABUmU3ulQh}
}
```

```
@inproceedings{
	zhang2022cgca,
	title={Probabilistic Implicit Scene Completion},
	author={Dongsu Zhang and Changwoon Choi and Inbum Park and Young Min Kim},
	booktitle={International Conference on Learning Representations},
	year={2022},
	url={https://openreview.net/forum?id=BnQhMqDfcKG}
}
```



# Acknowledgements

Our work is partially based on the open source work: [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine), [torch-scatter](https://github.com/rusty1s/pytorch_scatter), [deep sdf](https://github.com/facebookresearch/DeepSDF) [convolutional occupancy networks](https://github.com/autonomousvision/convolutional_occupancy_networks). We highly appreciate their contributions. 
