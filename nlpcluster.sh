mynlprun="nlprun3 -x jagupard[10-26] -a is -p high"


# Experiments with Linear Network Sweeps Over Dimension

# # Sweep over the dimension with alpha 1e-3
# CMD="${mynlprun} \"python run.py --config "exp1.yaml" --doc "exp_1_alpha_1e-3" --exp1=True --alpha=1e-3"\"
# eval ${CMD}
# sleep 1


# # Sweep over the dimension with alpha 1e-4
# CMD="${mynlprun} \"python run.py --config "exp1.yaml" --doc "exp_1_alpha_1e-4" --exp1=True --alpha=1e-4"\"
# eval ${CMD}
# sleep 1


# # Sweep over the dimension with alpha 0
# CMD="${mynlprun} \"python run.py --config "exp1.yaml" --doc "exp_1_alpha_0" --exp1=True --alpha=0.0"\"
# eval ${CMD}
# sleep 1

# # Sweep over the dimension with alpha 1e-2
# CMD="${mynlprun} \"python run.py --config "exp1.yaml" --doc "exp_1_alpha_1e-2" --exp1=True --alpha=1e-2"\"
# eval ${CMD}
# sleep 1


# Experiments with Linear Network Sweeps Over alpha

# Sweep over the alpha with dimension 200
# CMD="${mynlprun} \"python run.py --config "exp2.yaml" --doc "exp_2_dimension_200" --exp2=True --dimension=200"\"
# eval ${CMD}
# sleep 1


# # Sweep over the alpha with dimension 500
# CMD="${mynlprun} \"python run.py --config "exp2.yaml" --doc "exp_2_dimension_500" --exp2=True --dimension=500"\"
# eval ${CMD}
# sleep 1

# # Sweep over the alpha with dimension 1000
# CMD="${mynlprun} \"python run.py --config "exp2.yaml" --doc "exp_2_dimension_1000" --exp2=True --dimension=1000"\"
# eval ${CMD}
# sleep 1

# # Sweep over the alpha with dimension 2000
# CMD="${mynlprun} \"python run.py --config "exp2.yaml" --doc "exp_2_dimension_2000" --exp2=True --dimension=2000"\"
# eval ${CMD}
# sleep 1


# Experiments with Neural Network Sweeps Over alpha

# Sweep over the alpha with ReLUNets
CMD="${mynlprun} \"python run.py --config "exp3.yaml" --doc "exp_3" --exp3=True"\"
eval ${CMD}
sleep 1