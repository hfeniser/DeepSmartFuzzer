# RLforDL

# Generating an Extended Test Set For Deep Learning Models By Using Reinforcement Learning To Maximize Coverage

## Installation
### 1) Downloading All Source Code
```
git clone https://github.com/hasanferit/RLforDL.git
```


### 2) Install Gym Environments
```
pip install -e gym_dnn_test/
```

### 3) Install Dependecies
```
pip install -r requirements.txt
```

## Run MNIST-LENET Experiment
```
python mnist_lenet_experiment.py --lenet <lenet model number> --rl-algorithm <name of rl-algorithm>
```
Example:
```
python mnist_lenet_experiment.py --lenet 1 --rl-algorithm dqn
```

See available options:
```
python mnist_lenet_experiment.py --help
```

## Copyright Notice
RLforDL Copyright (C) 2019 Bogazici University

RLforDL is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

RLforDL is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with RLforDL. If not, see https://www.gnu.org/licenses/.

mail: hasan.eniser@boun.edu.tr, samet.demir1@boun.edu.tr

