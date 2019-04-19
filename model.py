import tensorflow as tf
import numpy as np
import cv2
train_gf_file_path='/media/fish/DATA/data/ILSVRC2012/ground_truth/train_annotation_full.txt'
val_gf_file_path='/media/fish/DATA/data/ILSVRC2012/ground_truth/val_annotation_full.txt'

IM_H=224
IM_W=224
TRAIN_BATCH_SIZE=2
LEARN_RATE=0.001

class SqueezeNet(object):
	def __init__(self, im_h, im_w, im_ch, cls_num, train_gf_fp,val_gf_fp,optimizer=tf.train.GradientDescentOptimizer):
	self.in_h = im_h
	self.in_w = im_w
	self.in_c = im_ch
	self.cls_num=cls_num
	self.optimizer=optimizer
	self.train_gf_fp=train_gf_fp
	self.val_gf_fp=val_gf_fp
		
	def gen_train_dataset(self):
		f=open(self.train_gf_fp)
		for line in f.readlines():
			lines=line.split(' ')
			label_idx=int(lines[1])
			label=np.zeros(self.cls_num)
			label[label_idx]=1.0
			im=cv2.imread(lines[0])
			im=cv2.resize(im,(self.im_w,self.im_h))
			im=im.astype('uint8')
			feature_data=np.zeros([self.im_h,self.im_w,self.in_c])
			feature_data=im
			yield (feature_data,label)


	def get_train_dataset(self):
		dataset=tf.data.Dataset.from_generator(self.gen_train_dataset,(tf.uint8,tf.int32),(tf.TensorShape([self.im_h,self.im_w,self.in_c]),tf.TensorShape([])))
		dataset=dataset.shuffle(1000).batch(TRAIN_BATCH_SIZE).repeat(10)
		return dataset

	def build_fire_module(self,input,ic,s1,e1,e3,fire_id):
		s1_filter=tf.Variable(tf.truncated_normal([1,1,ic,s1]))
		s1_bias=tf.Variable(tf.truncated_normal([s1]))
		e1_filter=tf.Variable(tf.truncated_normal([1,1,s1,e1]))
		e1_bias=tf.Variable(tf.truncated_normal([e1]))
		e3_filter=tf.Variable(tf.truncated_normal([3,3,s1,e3]))
		e3_bias=tf.Variable(tf.truncated_normal([e3]))
		with tf.name_scope(fire_id):
			output = tf.nn.conv2d(input, s1_filter, strides=[1,1,1,1], padding='SAME', name='conv_s_1')
			output = tf.nn.bias_add(output, s1_bias)
			expand1 = tf.nn.conv2d(output, e1_filter, strides=[1,1,1,1], padding='SAME', name='conv_e_1')
			expand1 = tf.nn.bias_add(expand1, e1_bias)
			expand3 = tf.nn.conv2d(output, e3_filter, strides=[1,1,1,1], padding='SAME', name='conv_e_3')
			expand3 = tf.nn.bias_add(expand3, e3_bias)	
			result = tf.concat([expand1, expand3], 3, name='concat_e1_e3')
			return tf.nn.relu(result)
			
	def build_net(self,input):
		conv1_ker=tf.Variable(tf.truncated_normal([7,7,3,96]))
		conv1_bias=tf.Variable(tf.random_normal([96]))
		output = tf.nn.conv2d(input, conv1_ker, strides=[1,2,2,1], padding='SAME', name='conv1')
		output = tf.nn.bias_add(output, conv1_bias)
		output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool1')
		output = self.build_fire_module(output,96,16,64,64,'fire2')
		output = self.build_fire_module(output,128,16,64,64,'fire3')
		output = self.build_fire_module(output,128,32,128,128,'fire4')
		output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool4')
		output = self.build_fire_module(output,256,32,128,128,'fire5')
		output = self.build_fire_module(output,256,48,192,192,'fire6')
		output = self.build_fire_module(output,384,48,192,192,'fire7')
		output = self.build_fire_module(output,384,64,256,256,'fire8')
		output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool8')
		output = self.build_fire_module(output,512,64,256,256,'fire9')
		conv10_ker=tf.Variable(tf.truncated_normal([1,1,512,1000]))
		conv10_bias=tf.Variable(tf.random_normal([1000]))	
		output = tf.nn.conv2d(output, conv10_ker, strides=[1,1,1,1], padding='SAME', name='conv10')
		output = tf.nn.bias_add(output, conv10_bias)
		output = tf.nn.avg_pool(output, ksize=[1, 13, 13, 1], strides=[1, 1, 1, 1], padding='SAME', name='avgpool10')
		avgpool10_shape=tf.shape(output)
		output=tf.reshape(output, [avgpool10_shape[0],-1])
		prob=tf.nn.softmax(output)
		return output,prob
	
	
if __name__ == '__main__':
	
	net=SqueezeNet(IM_H,IM_W,3,1000,train_gf_file_path,val_gf_file_path)
	dataset=net.get_train_dataset()

	inputs = tf.placeholder(tf.float32, shape=(None,IM_H,IM_W,3))
	labels = tf.placeholder(tf.float32, shape=(None,1000))
	
	logits_out,prob_out=net.build_net(inputs)
        loss_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_out, labels=train_labels))
        self.optimize = self.optimizer(LEARN_RATE).minimize(loss_cost)
	
	sess=tf.Session()
	train_ims,train_labels=get_train_dataset()
	iterator = dataset.make_initializable_iterator()
	features_elem,label_elem = iterator.get_next()
	print("ITERATOR",iterator)
	print("FEATURES",features_elem)
	print("LABELS",label_elem)
	


	sess=tf.Session()
	sess.run(iterator.initializer)
	feature_data=sess.run(features_elem)
	label_data=sess.run(label_elem)
	sess.run(iterator.initializer,feed_dict={features_placeholder: feature_data,labels_placeholder: label_data})
	sess.run()
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
