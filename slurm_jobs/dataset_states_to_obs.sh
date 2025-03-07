#!/bin/bash

#SBATCH -p accelerated
#SBATCH -n 1
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 4

#SBATCH --output=/home/hk-project-robolear/ll6323/project/fast_mail/slurm_jobs/output/%j.out

eval "$(conda shell.bash hook)"
conda activate atalay_mail

export LD_LIBRARY_PATH=$HOME/miniconda3/lib:$LD_LIBRARY_PATH
export CC=/opt/gcc/11/bin/gcc
export CXX=/opt/gcc/11/bin/g++
export CUB_HOME=$HOME/project/atalay_master_thesis/3D-Diffusion-Policy/third_party/cub-2.1.0
export CXXFLAGS="-O2 -march=core-avx2"
export CFLAGS="-O2 -march=core-avx2"

export OMP_NUM_THREADS=1
export MPI_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

export CAMERA_WIDTH=128
export CAMERA_HEIGHT=128

python -m robocasa.scripts.dataset_states_to_obs \
    --dataset /hkfs/work/workspace/scratch/ll6323-david_dataset_2/robocasa_datasets/v0.1/single_stage/kitchen_doors/CloseSingleDoor/2024-04-24/demo.hdf5 \
    --output processed_demo_${CAMERA_WIDTH}_${CAMERA_HEIGHT}.hdf5 \
    --camera_names robot0_agentview_left robot0_agentview_right robot0_agentview_center robot0_eye_in_hand \
    --camera_width ${CAMERA_WIDTH} \
    --camera_height ${CAMERA_HEIGHT}
