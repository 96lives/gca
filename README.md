# Code for Probabilistic Implicit Scene Completion (ICLR 2022)



![Alt text](media/teaser.jpg?raw=true "Title")

This repository contains code for reproducing the results for [probabilistic implicit scene completion](https://openreview.net/forum?id=BnQhMqDfcKG), which is accepted to ICLR 2022.
Currently, this repo contains training and testing results on ShapeNet sofa dataset for our model, named continuous Generative Cellular Automata (cGCA).

(2022/01/29) The paper has been accepted to ICLR 2022 as spotlight session. In the next several weeks, we are planning on moving the repo with further detailed instructions on how to reproduce the results. I'm also planning on merging the repo with our previous work [Generative Cellular Automata](https://openreview.net/forum?id=rABUmU3ulQh) (which this repo already contains). So stay tuned!! If you have any questions, please contact 96lives@snu.ac.kr :) 



## Installation & Data Preparation

1. **Anaconda and environment installations for training & testing**

```
conda create -n cgca python=3.8
conda activate cgca

# install torch
# for cuda 11.2
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
# for cuda 10.2
pip install torch==1.7.1 torchvision==0.8.2

# install mink v0.5.0 (versions above 0.5.0 are not tested and probabiliy won't work)
conda install openblas-devel -c anaconda 
git clone https://github.com/NVIDIA/MinkowskiEngine
cd MinkowskiEngine
pip install ./ -v --no-deps --install-option="--blas=openblas" --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" 

# install all other requirements
pip install -r requirements.txt
```
Please install [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) in version 0.5.0, otherwise the code might not work as expected.
The repo was tested with NVIDIA 2080ti GPU (11GB).


2. **Data preparation and Pretrained Models**

Download the [link](https://drive.google.com/file/d/1QnrPQQEeeasGmrcBf2yu0T2YvAVI8GqJ/view?usp=sharing) for ShapeNet sofa dataset and its preprocessed sparse voxel embedding.
The pretrained models are in the [link](https://drive.google.com/file/d/1qF-F2FWMMUWhtoYbq-ViqUPc5tGma58v/view?usp=sharing).
Place the files in the repo root.
The directory should look like the following:

```
\cgca
   - main.py
   ...
   - data/
      - cgca_shapenet/
         ...
      - embeddings/
         ...
   - pretrained_models/
      - sofa_transition
         ...
```

## Training 
![Alt text](media/method_overview.jpg?raw=true "Title")
1. Training autoencoder for sparse voxel embedding

To train the autoencoder model, run
```
python main --config configs/cgca_autoencoder-sofa.yaml -l log/autoencoder-sofa
```

2. Training the cGCA transition model

You do not need to train the autoencoder from scratch to train the transition model.
We have already preprocessed the ground truth sparse voxel embedding for the dataset in the above data preparation step.
The sparse voxel embeddings and pretrained autoencoder models (with configs) are in `data/embeddings/shapenet/sofa`
To train the transition model, run
```
python main --config configs/cgca_transition-sofa.yaml -l log/transition-sofa
```

3. Log visualization

The log files for the tensorboard visualization is available on the `log` directory.
To view the logs, run
```
tensorboard --logdir ./log
```
and enter the corresponding website with port on your web browser.


## Testing cGCA
Download the pretrained models as described in the above.
For the efficiency of vRAM usage, we test the cGCA in 2 step procedure.

1. Creating the last sparse voxel embedding 

This procedure iterates by transitions and caches the last sparse voxel embedding (s^{T+T'} in the main paper).
```
python main --test --resume-ckpt pretrained_models/sofa_transition/ckpts/ckpt-step-200000 --override "cache_only=True" -l log/sofa_test
```
2. Testing the metrics (this also generates meshes)

This procedure decodes the cached sparse voxel embedding s^{T+T'}.
You can find the reconstructed meshes (obj files) in the `log/sofa_test/test_save/step-200000/mesh/initial_mesh` folder.

```
python main --test --resume-ckpt pretrained_models/sofa_transition/ckpts/ckpt-step-200000 --override "cache_only=False" -l log/sofa_test
```



# Training GCA

We also added the code for baseline the original [Generative Cellular Automata](https://openreview.net/forum?id=rABUmU3ulQh) (GCA), published at ICLR 2021. You can train the model by running 

```
python main --config configs/gca-sofa.yaml -l log/gca-sofa
```



## Citation

If you find this repo useful for your research, please cite 
```
@inproceedings{
   anonymous2022probabilistic,
   title={Probabilistic Implicit Scene Completion},
   author={Anonymous},
   booktitle={Submitted to The Tenth International Conference on Learning Representations },
   year={2022},
   url={https://openreview.net/forum?id=BnQhMqDfcKG},
   note={under review}
}
```
