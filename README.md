# Benign Overfitting in Deep Linear Networks

![Excess Risk as the initialization variance is scaled](figures/scaling_alpha.png)
```
@article{
chatterji2022benigndeep,
title={Deep linear networks can benignly overfit when shallow ones do},
author={Niladri S. Chatterji and Philip M. Long},
journal={arXiv preprint arXiv:2209.09315},
year={2022}
}
```

- [Code](https://github.com/niladri-chatterji/benign-deep-linear)
- [arXiv](https://arxiv.org/abs/2209.09315)

## Setup instructions

1. Run `conda env create -f conda_env.yml`
2. Activate the conda environment by running `conda activate deeplinear`

## Reproducing experiments

The commandline scripts below will launch experiments in the paper.


### Figure 1 (Scaling the initialization variance in deep linear networks)


```bash
# Sweep over initialization scale of first layer weights alpha for different values of input dimension
python run.py --config "exp2.yaml" --doc "exp_2_dimension_500" --exp2=True --dimension=500
python run.py --config "exp2.yaml" --doc "exp_2_dimension_500" --exp2=True --dimension=1000
python run.py --config "exp2.yaml" --doc "exp_2_dimension_500" --exp2=True --dimension=2000

# Sweep over initialization scale of last layer weights beta for different values of input dimension
python run.py --config "exp4.yaml" --doc "exp_4_dimension_500" --exp4=True --dimension=500
python run.py --config "exp4.yaml" --doc "exp_4_dimension_500" --exp4=True --dimension=1000
python run.py --config "exp4.yaml" --doc "exp_4_dimension_500" --exp4=True --dimension=2000
```
### Figure 2 (Scaling the initialization variance in deep ReLU networks)


```bash
# Sweep over initialization scale of first layer weights alpha
python run.py --config "exp3.yaml" --doc "exp_3" --exp3=True

# Sweep over initialization scale of last layer weights beta
python run.py --config "exp5.yaml" --doc "exp_5" --exp5=True
```
### Figure 3 (Scaling the input dimension while holding the initialization scale fixed in deep linear networks)


```bash
# Sweep over input dimension d
python run.py --config "exp1.yaml" --doc "exp_1_alpha_0" --exp1=True --alpha=0.0
python run.py --config "exp1.yaml" --doc "exp_1_alpha_0" --exp1=True --alpha=1e-4
python run.py --config "exp1.yaml" --doc "exp_1_alpha_0" --exp1=True --alpha=1e-3
```
