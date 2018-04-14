import random
import os
import shutil
def generate_name(num):
	return '0'*(5-len(str(num)))+str(num)+'.png'

def get_dst_path():
	if not os.path.exists(os.path.join('GTA5', 'val')):
		os.mkdir(os.path.join('GTA5', 'val'))
	if not os.path.exists(os.path.join('GTA5', 'val','images')):
		os.mkdir(os.path.join('GTA5', 'val', 'images'))
	if not os.path.exists(os.path.join('GTA5', 'val', 'labels')):
		os.mkdir(os.path.join('GTA5', 'val', 'labels'))
		
	val_images = os.path.join('GTA5', 'val', 'images', filename)
	val_labels = os.path.join('GTA5', 'val', 'labels', filename)
	return val_images,val_labels

def get_src_path(filename):
	
	src_images = os.path.join('GTA5', 'images', filename)
	src_labels = os.path.join('GTA5', 'labels', filename)
	return src_images,src_labels

def mv_images(filename):
	src_images,src_labels = get_src_path(filename)
	val_images,val_labels = get_dst_path(filename)
	shutil.move("oldpos","newpos")    
for i in range(3000):
	file = random.randint(1,24966) 
	filename = generate_name(file)

