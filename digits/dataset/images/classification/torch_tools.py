import os
from random import shuffle
import shutil
import sys

def generate_dataset(jobs_folder, path, split):
	for root, dirs, files in os.walk(path):
		className = root.split("/")[-1:][0]
		shuffle(files)
		nbrImagesV = int((len(files) * split) / float(100)) #check again
		if len(className) > 0 and len(files) > 0:
			nPath = jobs_folder + "/train/" + className
			if not os.path.exists(nPath):
			        os.makedirs(nPath)

			for name in files[nbrImagesV:]:
			        if name.lower().endswith((".jpg", ".jpeg")):
			                shutil.copyfile(root + "/" + name, nPath + "/" + name)
			
			nPath = jobs_folder + "/val/" + className
			if not os.path.exists(nPath):
			        os.makedirs(nPath)

			for name in files[:nbrImagesV]:
			        if name.lower().endswith((".jpg", ".jpeg")):
			                shutil.copyfile(root + "/" + name, nPath + "/" + name)