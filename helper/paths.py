import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))

# Config yaml file
config_path = os.path.join(root_path, 'config/experiment.yaml')

# Data of the dataset
dataset_path = os.path.join(root_path, 'data/')