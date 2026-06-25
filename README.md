# LQDM

## Installation

Create a new conda environment, and install the dependencies:

```
conda create -n lqdm
conda activate lqdm
python3 -m pip install -r requirements.txt
```

## Data

Download the data and model checkpoint from [here](https://drive.google.com/drive/folders/1KIouI3V6XdIfXIgko9MYjBQfar183Wej?usp=sharing). Create a new directory `datasets/` from the project root, and place the ZIP file inside. Create a new directory `checkpoints/` from the project root, and place the checkpoint file inside.

## Running the code

### Training
```
python3 main.py fit -c configs/lqds.yaml --trainer.devices 4 --data.batch_size 4 --data.path
datasets/
```

This will train the model from scratch using the dataset from the config. Replace with 
the number of GPUs and the batch size based on available memory.

### Validation
```
python3 main.py validate -c configs/lqds.yaml --trainer.devices 1 --data.batch_size 4 --data.path datasets/ --model.ckpt_path checkpoints/model.ckpt
```

This will evaluate metrics for a trained model checkpoint.
