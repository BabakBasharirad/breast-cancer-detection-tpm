import os
import yaml

# Load and return configuration from YAML file
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config.yml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
