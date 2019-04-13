# RLforDL

# Generating an Extended Test Set For Deep Learning Models By Using Reinforcement Learning To Maximize Coverage

## Installation
### 1) Downloading All Source Code
```
git clone --recursive https://github.com/hasanferit/RLforDL.git
```


### 2) Install All
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