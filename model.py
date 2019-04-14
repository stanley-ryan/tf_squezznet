import tensorflow as tf
import numpy as np
import cv2
train_gf_file_path='/media/fish/DATA/data/ILSVRC2012/ground_truth/train_annotation_full.txt'
val_gf_file_path='/media/fish/DATA/data/ILSVRC2012/ground_truth/val_annotation_full.txt'

IM_H=224
IM_W=224
TRAIN_BATCH_SIZE=1
def gen_train_dataset():
	f=open(train_gf_file_path)
	for line in f.readlines():
		lines=line.split(' ')
		label=int(lines[1])
		im=cv2.imread(lines[0])
		im=cv2.resize(im,(IM_W,IM_H))
		yield (im,label)


def get_train_dataset():
	dataset=tf.data.Dataset.from_generator(gen_train_dataset,(tf.float32,tf.int32),(tf.TensorShape([IM_H,IM_W,3]),tf.TensorShape([])))
	dataset=dataset.batch(TRAIN_BATCH_SIZE)
	return dataset

if __name__ == '__main__':
	dataset=get_train_dataset()
'''
	iterr=dataset.make_one_shot_iterator()
	im_data,label_data=iterr.get_next()
	sess=tf.Session()
	example=sess.run(im_data)
	example=example.astype('uint8')
	im_disp=np.zeros([IM_H,IM_W,3])
	im_disp=example[0,:,:,:]
	print(example.dtype,'hehhe',example.shape,im_disp.shape)
	cv2.imshow('hahah',im_disp)
	cv2.waitKey(0)
	#print(example)
'''
