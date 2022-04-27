#/bin/bash

# PBS -l nodes=1:ppn=5,pmem=12GB

module load anaconda3
conda install -c anaconda pandas
conda install -c conda-forge typer
conda install -c conda-forge loguru

python3 train_reg_hyper_main_2.py 2021-06-20T23:00:00
