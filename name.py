import os

root = os.getenv('MODEL_PATH', "")
network_name = "red-mediana-profunda"
model_path = os.path.join(root, "models", network_name)
