import os
import sys

PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(PROJECT_PATH, "src")
MODULE_PATH = os.path.join(SOURCE_PATH, "tei_client")
sys.path.append(SOURCE_PATH)
sys.path.append(MODULE_PATH)
