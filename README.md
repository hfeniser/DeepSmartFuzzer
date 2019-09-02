# RLforDL

# Generating coverage-increasing mutated inputs by exploiting the explored coverage-increase patterns

### 1) Install Dependencies
```
pip install -r requirements.txt
```

### 2) MNIST-LeNet Experiments
* deephunter: runs deephunter
* mcts_batch: runs mcts on batches formed from the testset
* mcts_one: runs mcts on one random input from the testset
* mcts: runs mcts one input at a time
* mcts_selected_batches: runs mcts on the given batches(deephunter_x.npy files in data dir)

Example Usage:
```
python mnist_experiments/mcts_selected_batches.py --lenet 5 --coverage nbc
```

## Copyright Notice
RLforDL Copyright (C) 2019 Bogazici University

RLforDL is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

RLforDL is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with RLforDL. If not, see https://www.gnu.org/licenses/.

mail: hasan.eniser@boun.edu.tr, samet.demir1@boun.edu.tr

