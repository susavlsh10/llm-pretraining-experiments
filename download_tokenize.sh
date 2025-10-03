#!/bin/bash
#SBATCH --qos=regular           # queue (debug, regular)
#SBATCH --job-name=fineweb_tokenize  # Job name
#SBATCH --output=fineweb_tokenize_output_%j.txt          # Output file (%j is the job ID)
#SBATCH --error=fineweb_tokenize_error_%j.txt            # Error file
#SBATCH --time=16:00:00                 # Wall clock time limit
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --constraint=cpu        # Request CPUs 
#SBATCH --account=dasrepo               # Specify the project account
#SBATCH --mail-user=sls7161@tamu.edu
#SBATCH --mail-type=BEGIN,END,FAIL          # Notify about job

# Update environment variables
# Define environment variables for the container
ENV_VARS="export TRITON_CACHE_DIR=/pscratch/sd/s/susav/.triton_cache && \
export TRANSFORMERS_CACHE=/pscratch/sd/s/susav/transformers_cache && \
export MPLCONFIGDIR=/pscratch/sd/s/susav/matplotlib_config && \
export HF_HOME=/pscratch/sd/s/susav/huggingface/cache && \
export HF_DATASETS_TRUST_REMOTE_CODE=1 && \
export HOME=/global/homes/s/susav && \
export HF_TOKEN=INSERT_TOKEN && \
export PATH=\"\$HOME/.local/bin:\$PATH\""

# Set the container and working directory
CONTAINER="susavlsh10/mlsys:nvresearch-v1"  # Replace with your Docker image
WORKDIR="/global/homes/s/susav/workspace/llm-pretraining-experiments"  # Working directory inside the container
SCRIPT="python dev/data/fineweb.py"


# Print the script being executed to the output file
echo "Running the following script inside the container:"
echo "shifter --image=$CONTAINER bash -c \"cd $WORKDIR && $SCRIPT\""

# Run the script inside the container with all environment variables
shifter --image=$CONTAINER bash -c "$ENV_VARS && cd $WORKDIR && $SCRIPT"