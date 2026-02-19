## Information
- This code file consists of four parts
    1. collecting the dataset (Dataset.py)
    2. train initial skill models (SpirlCL.py)
    3. meta-train (Sisl.py)
    4. meta-test (MetaTest.py)
- Due to attachment size limitations, we are unable to provide the dataset directly. Instead, we have included code to collect the dataset. Please note that this may result in performance differences depending on the offline dataset used.
- This code file only provides the Kitchen, Maze2D, and AntMaze environments, and does not include the Office environment due to the attachment size limit.


## Requirements
- Python 3.8
- Pytorch 1.13.1
- Mujoco
- pip packages listed in requirements.txt


## 1. Collecting the dataset
Use Dataset.py to collect an offline dataset. For example:
```
python Dataset.py --env kitchen --noise 0.3 --max_transition 1000000 --device cpu

python Dataset.py --env maze --noise 0 --max_transition 500000 --device 0
```
After collection is complete, the dataset is stored in the ```./environments/{env}/dataset``` folder.  
Argument Description:
- ```env```: specify one of the following environments [“kitchen”, “maze”, “antmaze”]
- ```noise```: specify gaussian noise
- ```max_transition```: set the maximum number of transitions to collect
- ```device```: CPU or GPU number


## 2. Train initial skill models
Use SpirlCL.py to train the initial skill models(skill encoder, skill decoder(=low-level policy), skill prior). For example:
```
python SpirlCL.py --env kitchen --dataset_path {replace with your dataset path} --device 0
```
After train is complete, the skill models are stored in the ```./environments/{env}/skill``` folder.  
Argument Description:
- ```env```: specify one of the following environments [“kitchen”, “maze”, “antmaze”]
- ```dataset_path```: your dataset path
- ```device```: CPU or GPU number

## 3. Meta-train
Use Sisl.py to perform the meta-train. For example:
```
python Sisl.py --env kitchen --iteration 5000 --device 0 --device_sub 0 1 2 3 4 5 6 7 --h_buffer_size 3000 --skill_buffer_size 10000 --skill_n_priority 200 --skill_k_iter 1000 --skill_temp 1.0 --exp_buffer_size 100000 --exp_kld 0.001 --rnd_ext 5 --rnd_int 0.1 --dataset_path {replace with your dataset path} --skill_path {replace with your initial skill models path}

python Sisl.py --env kitchen --iteration 5000 --device 0 --device_sub 0 1 2 3 4 5 6 7 --dataset_path {replace with your dataset path} --skill_path {replace with your initial skill models path}

python Sisl.py --env maze --iteration 2000 --device 0 --device_sub 0 1 2 3 4 5 6 7 --dataset_path {replace with your dataset path} --skill_path {replace with your initial skill models path}

python Sisl.py --env antmaze --iteration 5000 --device 0 --device_sub 0 1 2 3 4 5 6 7 --dataset_path {replace with your dataset path} --skill_path {replace with your initial skill models path}
```
After train is complete, the sisl models are stored in the ```./environments/{env}/SISL``` folder.  
For each iteration, the control factor beta and the score per meta-train task are printed to the terminal.  
Argument Description:
- ```env```: specify one of the following environments [“kitchen”, “maze”, “antmaze”]
- ```iteration```: set the number of meta-train iteration
- ```device```: set the device to use as main for training (cpu or gpu number)
- ```device_sub```: set a device list to use as multiprocessing (if none, set the same value as ```device```)
- ```h_buffer_size```: high-level buffer size (per task, high-level transition) $\mathcal{B}_h$
- ```skill_buffer_size```: online buffer size (per task, low-level transition) $\mathcal{B}_\text{on}$
- ```skill_n_priority```: number of priority update trajectory $N_\text{priority}$
- ```skill_k_iter```: skill refinement $K_\text{iter}$
    - Note: Unlike in paper, we represent the rollout of high-level and exploration as one iteration, so we use half of the hyperparameters in paper as the default setup: kitchen=1000, office=1000, maze=500, antmaze=1000.
- ```skill_temp```: prioritization temperature $T$
- ```exp_buffer_size```: skill-improvement buffer size (per task, low-level transition) $\mathcal{B}_\text{imp}$
- ```exp_kld```: skill-improvement KLD coeff. $\lambda^\text{kld}_\text{imp}$
- ```rnd_ext```: RND extrinsic ratio $\delta_\text{ext}$
- ```rnd_int```: RND intrinsic ratio $\delta_\text{int}$
- ```dataset_path```: your dataset path
- ```skill_path```: your skill models path

## 4. Meta-test
Use MetaTest.py to perform the meta-test. For example:
```
python MetaTest.py --env kitchen --iteration 500 --device 0 --model_path {replace your sisl model path}
```
For each iteration, the score of the meta-test task is printed to the terminal.  
Argument Description:
- ```env```: specify one of the following four environments [“kitchen”, “office”, “maze”, “antmaze”]
- ```iteration```: set the number of meta-test iteration
- ```device```: CPU or GPU number
- ```model_path```: your sisl model path