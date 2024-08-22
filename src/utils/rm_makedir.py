import os
import shutil

def rm_makedir(path):
    if os.path.exists(path): shutil.rmtree(path)
    os.makedirs(path)