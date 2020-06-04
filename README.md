# Adaptive Reward-Free Exploration

This repository contains the source code used for the paper *Adaptive Reward-Free Exploration*.

## Installation

Install the required packages for Python 3 with:

```pip install requirements.txt```

## Instructions

To reproduce the experiments, simply run:

#### Figure 2.a
```
python3 estimation_error.py configs/double_chain.json
```

#### Figure 2.b
```
python3 state_occupancies.py configs/double_chain.json
```

#### Figure 2.c and 2.d
```
python3 sample_complexity.py configs/double_chain.json
```

#### Figure 3.a
```
python3 estimation_error.py configs/gridworld.json
```
#### Figure 3.b
```
python3 state_occupancies.py configs/gridworld.json
```

The results will appear in the `out` directory.