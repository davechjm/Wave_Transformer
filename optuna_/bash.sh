#!/usr/bin/env bash
#SBATCH --job-name=wave_t2
#SBATCH --output=py_torch_test%j.log
#SBATCH --error=py_torch_test%j.err
#SBATCH --mail-user=choi@uni-hildesheim.de
#SBATCH --partition=GPU,NGPU
#SBATCH --gres=gpu:1

cd python /home/choi/Wave_Transformer/optuna_     # navigate to the directory if necessary      

source activate pytorchenv
#srun python vanila_transformer_dwt_.py        # python jobs require the srun command to work
srun python DUTC_Test_RevIN2.py
