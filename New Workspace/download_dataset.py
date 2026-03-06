import os
import shutil
from dotenv import load_dotenv
from roboflow import Roboflow


load_dotenv()

API_KEY = os.getenv("ROBOFLOW_API_KEY")
WORKSPACE = os.getenv("WORKSPACE")
PROJECT = os.getenv("PROJECT")
VERSION = int(os.getenv("VERSION"))

print("Connecting to Roboflow...")

rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
version = project.version(VERSION)

print(f"Downloading dataset version {VERSION}...")

dataset = version.download("yolov8")

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "data")

if os.path.exists(data_path):
    shutil.rmtree(data_path)

shutil.move(dataset.location, data_path)

print("Dataset downloaded and moved to ./data successfully.")
