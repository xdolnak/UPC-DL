import sys,os
import random
import shutil
# Splits the images to 3 groups = training, validation and testing
def write_file(dest_path, split_group, type, file, path):
	type_path = dest_path + split_group + type
        if not os.path.exists(type_path):
        	os.makedirs(type_path)
        shutil.copy2(path + '/' + file, type_path + '/' + file)


min_image_cat=457 # classes with smaller number of image will be cut out
random.seed(230)
filenames = []
dirs = []
dest_path = '/home/nct01/nct01003/.keras/datasets/indoor/'
for path, subdirs, files in os.walk('/home/nct01/nct01003/.keras/datasets/Images/'):
	if (path != '/home/nct01/nct01003/.keras/datasets/Images/'):
		filenames = []
		for name in files:
			filenames.append(name)
		# sort
		print len(filenames), 'len'
		if (len(filenames) >= min_image_cat):
			filenames.sort()  # make sure that the filenames have a fixed order before shuffling
			filenames = random.sample(filenames, min_image_cat) # sample of the same size (deterministic given the chosen see)
			random.shuffle(filenames) # shuffles the ordering of filenames (deterministic given the chosen seed)
			# split
			split_1 = int(0.75 * len(filenames))
			split_2 = int(0.9 * len(filenames))
			train_filenames = filenames[:split_1]
			val_filenames = filenames[split_1:split_2]
			test_filenames = filenames[split_2:]
			# directory
			type = path.rsplit('/').pop()
			xprint type, 'type'
			# write train
			for file in train_filenames:
				write_file(dest_path, 'train/', type, file, path)
			#filenames.append(os.path.join(path, name))
        		#write val
			for file in val_filenames:
                                write_file(dest_path, 'val/', type, file, path)
			#write test
			for file in test_filenames:
                                write_file(dest_path, 'test/', type, file, path) 
