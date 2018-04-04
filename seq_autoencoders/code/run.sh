#!/bin/bash
#
#SBATCH --partition=titanx-long    # Partition to submit to <m40-short|m40-long|teslax-short|teslax-long>
#SBATCH --job-name=seq_autoencoder
#SBATCH -o seq_autoencoder_res_%j.txt            # output file
#SBATCH -e seq_autoencoder_res_%j.err            # File to which STDERR will be written
#SBATCH --ntasks=1
#SBATCH --time=04-01:00:00          # D-HH:MM:SS
#SBATCH --mem=50000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shasvatmukes@cs.umass.edu


module load python2/current
#source activate py27

python seqtoseq.py

#lspci -vnn|grep NVIDIA

#hostname

#python -u mnist-deep.py
#python -u /home/rgangaraju/moss-lab-git/lstm/batch_train_model.py
#python -u /home/rgangaraju/lstm/create_graph.py

sleep 1
exit

