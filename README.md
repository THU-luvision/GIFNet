# Code and dataset of *Group interaction field for learning and explaining pedestrian anticipation*

If you find it difficult to deploy/use/reproduce/modify, feel free to start an issue or [contact me](mailto:wang-xy22@mails.tsinghua.edu.cn).

If you want to reproduce the result in paper or use this code for other purpose, please follow the instructions below.



## Requirements and environment

1. python 3, tested on ubuntu 20.04.
2. install PyTorch following the [official instruction](https://pytorch.org/).
3. install other dependencies.

```shell
pip install tensorboard tensorboardx matplotlib seaborn scikit-learn
```



## Dataset

The dataset used for training and evaluating GIFNet, and supporting the plots and other findings of this study is generated based on the  [PANDA dataset](http://www.panda-dataset.com/). Note that we did not use the original raw dataset, because our GIFNet only requires for the inputs of trajectory sequence and visual orientation sequence of pedestrians, as well as group and interaction state information. In other words, after the particular pre-processing, the dataset used in our work has nothing to do with the identifiable information such as face and appearance that may be related to pedestrians' personal privacy. 

PANDA consists of 21 real outdoor scenes with a diversity of scenarios, pedestrians' density, trajectory distribution and group activity. We select 8 representative scenes with rich pedestrians. Each scene consists of approximately 3600 frames (approximately two minutes). In total, there are 21704 trajectory sequences. The training, testing and validation sets contain 15,511, 3052, and 3,141 sequences, respectively. 

The dataset contains the sequences of location and visual orientation of pedestrians in world coordinates, as well as group and interaction state information. In the pre-processing of the PANDA dataset, we compute a homography matrix to map images from the origin view to the top view for each scene, and we apply homography transformation to transpose the coordinates of locations in pixels to world coordinates in metres.

The dataset can be download from [here](https://drive.google.com/file/d/1LXzA3EeJLqK2veqZ_bVgw7C6BrJYhB9E/view?usp=sharing). Please refer to the [How to run](https://github.com/THU-luvision/GIFNet#how-to-run) for the usage.

### The dataset file structure

```
- <GIF_Dataset>
    - poi_trajectory.npy: trajectory sequences of all pedestrian of interest
    - neighbour_trajectory.npy: trajectory sequences of all corresponding neighbour
    - poi_viusal_orientation.npy: visual orientation sequences of all pedestrian of interest
    - neighbour_viusal_orientation.npy: visual orientation sequences of all corresponding neighbour
    - evaluate_z_20.npy: fixed Gaussian noise to produce the proxemics fields
    - <info_dicts>: json files containing information about group neighbours and interaction states of each sequence
    - <checkpoints>: trained GIFNet model parameters for predicting the proxemics field and the attention field
    - <predictions>: Sample GIFNet predictions and groundtruths of the proxemics field and the attention field
```



## How to run

### Clone the repository

```shell
git clone https://github.com/THU-luvision/GIFNet.git
cd GIFNet
```

### File structure

```
- <Root>
    - <dataset>: folder to place released dataset
    - <data>: folder containing source codes for data pre-processing
    - models.py: PyTorch models for GIFNet
    - train_proxemics_field.py: code for training GIFNet for predicting the proxemics field
    - train_attention_field.py: code for training GIFNet for predicting the attention field
    - train_proxemics_wo_GIF-GAT.py: code for training GIFNet (without GIF-GAT) for predicting the proxemics field
    - train_attention_wo_GIF-GAT.py: code for training GIFNet (without GIF-GAT) for predicting the attention field
    - evaluate_proxemics_field.py: code for evaluating trained GIFNet for predicting the proxemics field
    - evaluate_attention_field.py: code for evaluating trained GIFNet for predicting the attention field
    - evaluate_proxemics_wo_GIF-GAT.py: code for evaluating trained GIFNet (without GIF-GAT) for predicting the proxemics field
    - evaluate_attention_wo_GIF-GAT.py: code for evaluating trained GIFNet (without GIF-GAT) for predicting the attention field
    - visualize_field.py: code for visualizing the proxemics field and the attention field
    - utils.py: other utilities
    - <robot_simulation> codebase for robot simulations
```

### Download the dataset

Download the dataset from [this link](https://drive.google.com/file/d/1LXzA3EeJLqK2veqZ_bVgw7C6BrJYhB9E/view?usp=sharing) and place the zip file into `dataset` folder, then unzip the folder as follows. 

```shell
mkdir dataset
cd dataset
unzip GIF_Dataset
cd ..
```

### Train the model

```shell
# Train the GIFNet for the proxemics field prediction
python train_proxemics_field.py

# Train the GIFNet for the attention field prediction
python train_attention_field.py
```

See the code to understand all the arguments that can be given to the command.

#### Evaluate the model & Do inference

```shell
# Evaluate trained GIFNet for the proxemics field prediction
python evaluate_proxemics_field.py

# Evaluate trained GIFNet for the attention field prediction
python evaluate_attention_field.py
```

Using the default parameters in the code and using the provided model weights (in `GIF_Dataset/checkpoints`), you can get experimental results presented in the paper. You can also use the evaluation code to do inference on the dataset and save the predictions.

### Visualize the proxemics field and the attention field

Using visualize_field.py to visualize the proxemics field and the attention field. The default is to use the predicted results in `GIF_Dataset/predictions`.

### Robot navigation simulations

```shell
cd robot_simulation
cd robot_navigation

# Run the simulation in Supplementary Fig. 1f&g
python simulate_world.py
python gen_results.py

# Run the simulation in Supplementary Fig. 1e
python simulate_disturb.py
```

Then we paste the output values into the MATLAB scripts in `plot_matlab` folder to get the sub-figures in the Supplementary Fig. 1.



## Citation

If you find this repository useful for your research or use our datasets, please cite the following paper:
```
@inproceedings{wang2020panda, title={PANDA: A Gigapixel-level Human-centric Video Dataset}, author={Wang, Xueyang and Zhang, Xiya and Zhu, Yinheng and Guo, Yuchen and Yuan, Xiaoyun and Xiang, Liuyu and Wang, Zerun and Ding, Guiguang and Brady, David J and Dai, Qionghai and Fang, Lu}, booktitle={Computer Vision and Pattern Recognition (CVPR), 2020 IEEE International Conference on}, year={2020}, organization={IEEE}}

```

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" href="http://purl.org/dc/dcmitype/Text" property="dct:title" rel="dct:type">GIFNet and dataset</span> are licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
