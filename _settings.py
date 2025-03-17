import getpass
import os
import sys

__USERNAME = getpass.getuser()

# Chante to your dir
_BASE_DIR = f'/home/local/ASURITE/tchen169/'

DATA_FOLDER = os.path.join(_BASE_DIR, 'NLGUQ')
GENERATION_FOLDER = os.path.join(DATA_FOLDER, 'output')
os.makedirs(GENERATION_FOLDER, exist_ok=True)


