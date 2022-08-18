# Benign Overfitting in Deep Linear Networks

![Excess Risk as the initialization variance is scaled](figures/scaling_alpha.png)
```
@article{TBD
}
```

- [Code](https://github.com/niladri-chatterji/Benign-Overfitting-in-Deep-Linear-Networks)
- [arXiv](TBD)

## Setup instructions

1. Run `conda env create -f conda_env.yml`
2. Activate the conda environment by running `conda activate deeplinear`

## Reproducing experiments

The commandline scripts below will launch experiments in the paper.


### Figure 1

Run `notebooks/two-gaussians.ipynb`

### Figure 2 (Importance Weighted Cross-Entropy Loss and VS Loss)


```bash
# Importance Weighted Cross-Entropy Loss Minority Samples Sweep
python run.py +experiment=cifar_reweighted_early_stopped_scaling trainer.max_epochs=800 datamodule.class_samples=[2500,500]
python run.py +experiment=cifar_reweighted_early_stopped_scaling trainer.max_epochs=800 datamodule.class_samples=[2500,1000]
python run.py +experiment=cifar_reweighted_early_stopped_scaling trainer.max_epochs=800 datamodule.class_samples=[2500,1500]
python run.py +experiment=cifar_reweighted_early_stopped_scaling trainer.max_epochs=800 datamodule.class_samples=[2500,2000]

# Importance Weighted Cross-Entropy Loss Majority Samples Sweep
python run.py +experiment=cifar_reweighted_early_stopped_scaling trainer.max_epochs=800 datamodule.class_samples=[3000,500]
python run.py +experiment=cifar_reweighted_early_stopped_scaling trainer.max_epochs=800 datamodule.class_samples=[3500,500]
python run.py +experiment=cifar_reweighted_early_stopped_scaling trainer.max_epochs=800 datamodule.class_samples=[4000,500]
python run.py +experiment=cifar_reweighted_early_stopped_scaling trainer.max_epochs=800 datamodule.class_samples=[4500,500]

# Importance Weighted Cross-Entropy Loss Propotional Increase Sweep
python run.py +experiment=cifar_reweighted_early_stopped_scaling trainer.max_epochs=800 datamodule.class_samples=[3000,600]
python run.py +experiment=cifar_reweighted_early_stopped_scaling trainer.max_epochs=800 datamodule.class_samples=[3500,700]
python run.py +experiment=cifar_reweighted_early_stopped_scaling trainer.max_epochs=800 datamodule.class_samples=[4000,800]
python run.py +experiment=cifar_reweighted_early_stopped_scaling trainer.max_epochs=800 datamodule.class_samples=[4500,900]
python run.py +experiment=cifar_reweighted_early_stopped_scaling trainer.max_epochs=800 datamodule.class_samples=[5000,1000]

# Importance Weighted VS Loss Minority Samples Sweep
python run.py +experiment=cifar_reweighted_early_stopped_scaling_vsloss trainer.max_epochs=800 datamodule.class_samples=[2500,500]
python run.py +experiment=cifar_reweighted_early_stopped_scaling_vsloss trainer.max_epochs=800 datamodule.class_samples=[2500,1000]
python run.py +experiment=cifar_reweighted_early_stopped_scaling_vsloss trainer.max_epochs=800 datamodule.class_samples=[2500,1500]
python run.py +experiment=cifar_reweighted_early_stopped_scaling_vsloss trainer.max_epochs=800 datamodule.class_samples=[2500,2000]

# Importance Weighted VS Loss Majority Samples Sweep
python run.py +experiment=cifar_reweighted_early_stopped_scaling_vsloss trainer.max_epochs=800 datamodule.class_samples=[3000,500]
python run.py +experiment=cifar_reweighted_early_stopped_scaling_vsloss trainer.max_epochs=800 datamodule.class_samples=[3500,500]
python run.py +experiment=cifar_reweighted_early_stopped_scaling_vsloss trainer.max_epochs=800 datamodule.class_samples=[4000,500]
python run.py +experiment=cifar_reweighted_early_stopped_scaling_vsloss trainer.max_epochs=800 datamodule.class_samples=[4500,500]

# Importance Weighted VS Loss Propotional Increase Sweep
python run.py +experiment=cifar_reweighted_early_stopped_scaling_vsloss trainer.max_epochs=800 datamodule.class_samples=[3000,600]
python run.py +experiment=cifar_reweighted_early_stopped_scaling_vsloss trainer.max_epochs=800 datamodule.class_samples=[3500,700]
python run.py +experiment=cifar_reweighted_early_stopped_scaling_vsloss trainer.max_epochs=800 datamodule.class_samples=[4000,800]
python run.py +experiment=cifar_reweighted_early_stopped_scaling_vsloss trainer.max_epochs=800 datamodule.class_samples=[4500,900]
python run.py +experiment=cifar_reweighted_early_stopped_scaling_vsloss trainer.max_epochs=800 datamodule.class_samples=[5000,1000]
```

### Figure 3 (Hat Function)

Run `notebooks/two-gaussians.ipynb`


### Figure 4 (Tilted loss and Group DRO)

```bash
# Tilted Loss Minority Samples Sweep
python run.py +experiment=cifar_scaling_minority_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[2500,500]
python run.py +experiment=cifar_scaling_minority_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[2500,1000]
python run.py +experiment=cifar_scaling_minority_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[2500,1500]
python run.py +experiment=cifar_scaling_minority_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[2500,2000]

# Tilted Loss Majority Samples Sweep
python run.py +experiment=cifar_scaling_majority_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[3000,500]
python run.py +experiment=cifar_scaling_majority_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[3500,500]
python run.py +experiment=cifar_scaling_majority_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[4000,500]
python run.py +experiment=cifar_scaling_majority_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[4500,500]

# Tilted Loss Propotional Increase Sweep
python run.py +experiment=cifar_scaling_n_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[3000,600]
python run.py +experiment=cifar_scaling_n_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[3500,700]
python run.py +experiment=cifar_scaling_n_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[4000,800]
python run.py +experiment=cifar_scaling_n_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[4500,900]
python run.py +experiment=cifar_scaling_n_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[5000,1000]

# Group DRO Minority Samples Sweep
python run.py +experiment=cifar_early_stopped_scaling_dro trainer.max_epochs=800 datamodule.class_samples=[2500,500]
python run.py +experiment=cifar_early_stopped_scaling_dro trainer.max_epochs=800 datamodule.class_samples=[2500,1000]
python run.py +experiment=cifar_early_stopped_scaling_dro trainer.max_epochs=800 datamodule.class_samples=[2500,1500]
python run.py +experiment=cifar_early_stopped_scaling_dro trainer.max_epochs=800 datamodule.class_samples=[2500,2000]

# Group DRO Majority Samples Sweep
python run.py +experiment=cifar_early_stopped_scaling_dro trainer.max_epochs=800 datamodule.class_samples=[3000,500]
python run.py +experiment=cifar_early_stopped_scaling_dro trainer.max_epochs=800 datamodule.class_samples=[3500,500]
python run.py +experiment=cifar_early_stopped_scaling_dro trainer.max_epochs=800 datamodule.class_samples=[4000,500]
python run.py +experiment=cifar_early_stopped_scaling_dro trainer.max_epochs=800 datamodule.class_samples=[4500,500]

# Group DRO Propotional Increase Sweep
python run.py +experiment=cifar_early_stopped_scaling_dro trainer.max_epochs=800 datamodule.class_samples=[3000,600]
python run.py +experiment=cifar_early_stopped_scaling_dro trainer.max_epochs=800 datamodule.class_samples=[3500,700]
python run.py +experiment=cifar_early_stopped_scaling_dro trainer.max_epochs=800 datamodule.class_samples=[4000,800]
python run.py +experiment=cifar_early_stopped_scaling_dro trainer.max_epochs=800 datamodule.class_samples=[4500,900]
python run.py +experiment=cifar_early_stopped_scaling_dro trainer.max_epochs=800 datamodule.class_samples=[5000,1000]
```
