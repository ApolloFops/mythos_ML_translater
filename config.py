import os

from scripts.tools.utility import getCachePath

PLUGIN_PATH = os.path.dirname(os.path.realpath(__file__))
PLUGIN_CACHE_PATH = getCachePath("mythos_ML_translater")

# ===== CONFIG =====
# Model and tokenizer settings
MODEL_NAME = "google/flan-t5-base" # Hugging Face model name; supports seq2seq tasks
DATA_FILE = PLUGIN_PATH + "/data.json" # Local JSON dataset file
MODEL_PATH = PLUGIN_CACHE_PATH + "/model" # Directory to save model & tokenizer

LOG_COMPONENT = "MythosML"

DATABASE_PATH = PLUGIN_CACHE_PATH + "/translations.db"
