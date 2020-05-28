# Adaptive Reward-Free Exploration

## Installation

```pip install requirements.txt```

## Instructions

To reproduce the experiments, simply run:
```
python3 estimation_error.py configs/double_chain.json
python3 state_occupancies.py configs/double_chain.json

python3 estimation_error.py configs/gridworld.json
python3 state_occupancies.py configs/gridworld.json

python3 sample_complexity.py configs/double_chain_exp_small.json
```

The results will appear in the `out` directory.