#!/bin/bash
#SBATCH --job-name=irds_exp
#SBATCH --mail-type=ALL
#SBATCH --mail-user=milena.mazza3@studio.unibo.it
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=31G
#SBATCH --partition=l40        # Usa 'l40' se vuoi le GPU migliori
#SBATCH --output=output_femnist.log
#SBATCH --chdir=/scratch.hpc/mazza/femnist/femnist_lab
#SBATCH --gres=gpu:1

# Attiva l'ambiente virtuale
source venv/bin/activate

# Avvia lo script
bash start_experiment.sh

