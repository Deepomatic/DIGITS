import os
from random import shuffle
import shutil
import sys

def generate_dataset(jobs_folder, path, split):
	nbrFiles = 0
	for root, dirs, files in os.walk(path):
		nbrFiles += len(files)

	count = 0
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
					print count, nbrFiles
					count += 1
			
			nPath = jobs_folder + "/val/" + className
			if not os.path.exists(nPath):
			        os.makedirs(nPath)

			for name in files[:nbrImagesV]:
				if name.lower().endswith((".jpg", ".jpeg")):
					shutil.copyfile(root + "/" + name, nPath + "/" + name)
					print count, nbrFiles
					count += 1

if __name__ == "__main__":
	##jobs_folder, path, split
	generate_dataset(sys.argv[1], sys.argv[2], sys.argv[3])
	print "done"