from read_datasets import pretreat_data
import tensorflow as tf
ckpt_dir = '/home/wu/ml_workspace/Multi_CNN_smog/model/mod04'
sess=tf.InteractiveSession()
result = pretreat_data(0,6,2,6)
data_cd = result

def conv2d(x,w): 
	return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME') 

def max_pool_2x2(x): 
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') 

def max_pool_2x2_large(x): 
	return tf.nn.max_pool(x,ksize=[1,5,5,1],strides=[1,5,5,1],padding='SAME') 


x=tf.placeholder(tf.float32,[None,80,90]) 
y_=tf.placeholder(tf.float32,[None,6]) 
x_img = tf.reshape(x,[-1,80,90,1])

w_conv1=tf.Variable(tf.truncated_normal([5,5,1,4],stddev=0.1)) 
b_conv1=tf.Variable(tf.constant(0.1,shape=[4])) 
h_conv1=tf.nn.relu(conv2d(x_img,w_conv1)+b_conv1) 
h_pool1=max_pool_2x2_large(h_conv1) 

w_conv2=tf.Variable(tf.truncated_normal([3,3,4,8],stddev=0.1)) 
b_conv2=tf.Variable(tf.constant(0.1,shape=[8])) 
h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2) 
h_pool2=max_pool_2x2(h_conv2) 

w_conv3=tf.Variable(tf.truncated_normal([3,3,8,16],stddev=0.1)) 
b_conv3=tf.Variable(tf.constant(0.1,shape=[16])) 
h_conv3=tf.nn.relu(conv2d(h_pool2,w_conv3)+b_conv3) 
h_pool3=max_pool_2x2(h_conv3) 

w_conv4=tf.Variable(tf.truncated_normal([3,3,16,32],stddev=0.1)) 
b_conv4=tf.Variable(tf.constant(0.1,shape=[32])) 
h_conv4=tf.nn.relu(conv2d(h_pool3,w_conv4)+b_conv4) 
h_pool4=max_pool_2x2(h_conv4)

w_fc1=tf.Variable(tf.truncated_normal([2*3*32,1024],stddev=0.1)) 
b_fc1=tf.Variable(tf.constant(0.1,shape=[1024])) 
h_pool4_flat=tf.reshape(h_pool4,[-1,2*3*32]) 
h_fc1=tf.nn.relu(tf.matmul(h_pool4_flat,w_fc1)+b_fc1) 

keep_prob=tf.placeholder(tf.float32) 
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob) 

w_fc2=tf.Variable(tf.truncated_normal([1024,6],stddev=0.1)) 
b_fc2=tf.Variable(tf.constant(0.1,shape=[6])) 
y_out=tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2) 

loss=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_out),reduction_indices=[1])) 

train_step=tf.train.AdamOptimizer(1e-4).minimize(loss) 

correct_prediction=tf.equal(tf.argmax(y_out,1),tf.argmax(y_,1)) 
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) 

tf.global_variables_initializer().run() 

saver = tf.train.Saver()
for i in range(10000):  
	batch=data_cd.train.next_batch(100) 
	if i%50==0: 
		train_accuracy=accuracy.eval(feed_dict={x:data_cd.test.images,y_:data_cd.test.labels,keep_prob:1}) 
		print("step %d,train_accuracy= %g"%(i,train_accuracy)) 
	train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5}) 

saver.save(sess,ckpt_dir + '/model.ckpt')
print("test_accuracy= %g"%accuracy.eval(feed_dict={x:data_cd.test.images,y_:data_cd.test.labels,keep_prob:1}))









