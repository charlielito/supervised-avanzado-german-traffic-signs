import os

root = os.getenv('MODEL_PATH', "")
network_name = "red-all-cnn"
model_path = os.path.join(root, "models", network_name)
