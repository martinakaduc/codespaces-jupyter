# Active learning with Python

This repository implements a few active learning algorithms in Python. The algorithms are implemented in a modular way, so that they can be used with any classifier. The algorithms are:
* Random sampling
* Uncertainty sampling
* Thomson sampling

```
python main.py --algo [ALGO] --exp_id [EXP_ID] --seeds [SEEDS] [--allow_replacement]
python plot.py --exp_id [EXP_ID]
```