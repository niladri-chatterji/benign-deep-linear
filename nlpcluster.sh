mynlprun="nlprun3 -x jagupard[10-26] -a is"


# Experiments with Linear Network Sweeps Over Dimension

# Sweep over the dimension with alpha 1e-3
CMD="${mynlprun} \"python run.py --config "exp1.yaml" --doc "exp_1_alpha_1e-3" --exp1=True --alpha=1e-3"\"
eval ${CMD}
sleep 1


# Sweep over the dimension with alpha 1e-4
CMD="${mynlprun} \"python run.py --config "exp1.yaml" --doc "exp_1_alpha_1e-4" --exp1=True --alpha=1e-4"\"
eval ${CMD}
sleep 1


# Sweep over the dimension with alpha 0
CMD="${mynlprun} \"python run.py --config "exp1.yaml" --doc "exp_1_alpha_0" --exp1=True --alpha=0.0"\"
eval ${CMD}
sleep 1

# Sweep over the dimension with alpha 1e-2
CMD="${mynlprun} \"python run.py --config "exp1.yaml" --doc "exp_1_alpha_1e-2" --exp1=True --alpha=1e-2"\"
eval ${CMD}
sleep 1