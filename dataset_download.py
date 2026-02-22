import os
from dotenv import load_dotenv
from roboflow import Roboflow
import shutil

load_dotenv()

API_KEY = os.getenv("ROBOFLOW_API_KEY")
WORKSPACE = os.getenv("WORKSPACE")
PROJECT = os.getenv("PROJECT")
VERSION = int(os.getenv("VERSION"))

rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
version = project.version(VERSION)

dataset = version.download("yolov12")

# Move downloaded folder to ./data
if os.path.exists("data"):
    shutil.rmtree("data")

shutil.move(dataset.location, "data")

print("Dataset downloaded and moved to ./data successfully.")
