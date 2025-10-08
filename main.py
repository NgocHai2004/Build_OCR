import yaml
import torch
from model import load_model
from inference import Run_inference

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

folder_path = config["folder_path"]
model_path = config["model_path"]
img_size = config["img_size"]
folder_output = config["folder_output"]

device = "cuda" if torch.cuda.is_available() else "cpu"

model = load_model(model_path, device)

Run_inference(model, folder_path, folder_output, device)
