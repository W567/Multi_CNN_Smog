
import numpy as np
import csv
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.contrib.learn.python.learn.datasets import base
import tifffile as tif 
import os
import random

"""
# read csv file and store all data (as uint8 datatype) in an array
"""
def read_csv_int(file_name):
	with open(file_name, newline = '') as f:
		reader = csv.reader(f)
		try:
			data = np.array([row for row in reader])
			data = data.astype('uint8')
			return data
		except csv.Error as e:
			sys.exit('file {}, line {}: {}'.format(filename, reader.line_num, e))

"""
# read csv file and store all data (as float32 datatype) in an array
"""
def read_csv_float(file_name):
  with open(file_name, newline = '') as f:
    reader = csv.reader(f)
    try:
      data = np.array([row for row in reader])
      data = data.astype('float32')
      return data
    except csv.Error as e:
      sys.exit('file {}, line {}: {}'.format(filename, reader.line_num, e))

"""
# read data from the specified row of a csv file and store the data (as uint8 datatype) in an array
"""
def read_csv_row(file_name, n_row):
	with open(file_name, newline = '') as f:
		reader = csv.reader(f)
		try:
			data = np.array([row for i,row in enumerate(reader) if i == n_row])
			data = data.astype('uint8')
			return data[0]
		except csv.Error as e:
			sys.exit('file {}, line {}: {}'.format(filename, reader.line_num, e))

"""
# Convert class labels from scalars to one-hot vectors.
"""
def dense_to_one_hot(labels_dense, num_classes):
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))  # dtype = int  改为int型, 默认输出float
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

"""
# default method for reading labels from a csv file
# not used in this project
"""
def read_labels(f, one_hot=False, num_classes=6):
  print('Reading labels')
  labels = read_csv_int(f)
  if one_hot:
    return dense_to_one_hot(labels, num_classes)
  return labels

"""
# read tiff file and store the data in an array
# input: file name of the tiff file
# output: an array with data stored
"""
def read_tiff(f):
  with tif.TiffFile(f) as tiff:
    image_stack = tiff.asarray()
    return image_stack

"""
# default DataSet Class Definition (from Tensorflow Tutorial for MNIST)
# with part for fake data deleted 
"""
class DataSet(object):

  def __init__(self,
               images,
               labels,
               one_hot=False,
               seed=None):

    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    np.random.seed(seed1 if seed is None else seed2)

    assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
    self._num_examples = images.shape[0]
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, shuffle=True):
    """Return the next `batch_size` examples from this data set."""

    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]

    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return np.concatenate((images_rest_part, images_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)

    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]



def read_datasets(mode,
                  skip_first_line = False,
                  test = True,
                  asArray = True):
  """
  " input: mode - set the mode to decide which type of images to read. { 0 : mod02_tif, 1 : mod02_rgb, 2 : mod04}
  "        skip_first_line - skip the first line or not when reading the labels csv files (default: True)
  "        asArray - store images and labels in arrays(True) or lists(False) (default: True)
  " output: images and corresponding labels stored in two lists or numpy.arrays, respectively
  "
  """
  mod02_dir = '/home/wu/ml_workspace/Multi_CNN_smog/datasets/mod02_tiff'  # mode = 0
  rgb_dir = '/home/wu/ml_workspace/Multi_CNN_smog/datasets/mod02_rgb/winter'     # mode = 1
  mod04_dir = '/home/wu/ml_workspace/Multi_CNN_smog/datasets/mod04'       # mode = 2
  img_dir = [mod02_dir, rgb_dir, mod04_dir]

  test_mod02_dir = '/home/wu/ml_workspace/Multi_CNN_smog/datasets/2016_tiff_no_cloud'
  test_rgb_dir   = '/home/wu/ml_workspace/Multi_CNN_smog/datasets/2016_rgb_no_cloud/winter'
  test_mod04_dir = '/home/wu/ml_workspace/Multi_CNN_smog/datasets/2016_mod04_no_cloud'
  test_dir = [test_mod02_dir, test_rgb_dir, test_mod04_dir]

  label_dir = '/home/wu/ml_workspace/Multi_CNN_smog/datasets/label'

  print('read_datasets')
  images = []          # images
  labels = []          # corresponding labels
  dir_temp = img_dir[mode] if not test else test_dir[mode]
  fname_list = os.listdir(dir_temp)  # search for image files
  fname_list = sorted(fname_list)         # sort the files according to their names (label information is stored in date order)
  num_f = len(fname_list)                 # count the number of files
  
  lname_list = os.listdir(label_dir) # search for label files
  lname_list = sorted(lname_list)    # sort the files according to their names (label information for each city is stored seperately in several csv files)

  for fname in fname_list:           # in the order of date, read one image file and its correponding label
    fpath = os.path.join(dir_temp, fname)     
    data = read_tiff(fpath)
    if len(data.shape) > 2:
      if data.shape[1]<180 or data.shape[2]<240 :     # ignore the image if its size is smaller than 180*240
        num_f = num_f -1 
        continue
      elif data.shape[1]>180 or data.shape[2]>240 :	  # normalize the size of image to 180*240 if its size is bigger than 180*240 
        data = data[:,0:180,0:240]    
    else:
      if data.shape[0]<180 or data.shape[1]<240 :     # ignore the image if its size is smaller than 180*240
        num_f = num_f -1 
        continue
      elif data.shape[0]>180 or data.shape[1]>240 :   # normalize the size of image to 180*240 if its size is bigger than 180*240 
        data = data[0:180,0:240]   

      for i in range(data.shape[0]):
        for j in range(data.shape[1]):
          if data[i][j] < 0:
            data[i][j] = 0
          elif type(data[i][j]) is not np.int16:
            print(i,j,data[i][j],type(data[i][j]))
            data[i][j] = 0 
          else:
            data[i][j] /= 500   

    images.append(data)

    label_d = []
    _date = fname.split(".")[0]                                                  # get the date information from the file name
    _row = int(_date.split("_")[1])+365*(int(_date.split("_")[0])-2016) + (int(_date.split("_")[0]) > 2016)   # get the row number from the file name
    for i in range(len(lname_list)):                                             # for each image, get the corresponding labels for all cities.
      lpath = os.path.join(label_dir,lname_list[i])

      label = read_csv_row(lpath, (_row - (not skip_first_line)))                 # be careful about the head line
      label_d.append(label)

      # print((_row - (not skip_first_line)), label)
    labels.append(label_d)

  if asArray:
    images = np.array(images)
    labels = np.array(labels)
  return images, labels


"""
# balance the number of datasets for each class by copying and adding data to the minority class
# input: images 
#        labels
# output: balanced images and corresponding labels (both as Array)
# reference: Hikvision project for ImageNet2016 - algorithm name: Label Shuffling
"""
def balance_label(images, labels):

  print('balance_label')
  if type(labels) is not list:         # set the type of labels to list
    labels = list(labels)
  if type(images) is not np.ndarray:   # set the type of images to array
    images = np.array(images)
  shape = list(images.shape)           # get the original images shape information and store in a list 
  class_num = len(set(labels))         # count the number of classes for the input images

  class_image = []
  class_count = []
  class_label = []
  for i in set(labels):               # for each class, get the quantity of that class and index in the list
    count_i = labels.count(i)
    class_count.append(count_i)          # get the quantity
    pos = -1
    index = []
    for j in range(count_i):             # get the index
      pos = labels.index(i, pos+1)
      index.append(pos)
    image_temp = [images[i] for i in index]  # classify all images according to their corresponding labels
    class_image.append(image_temp)
    class_label.append(i)                    # store the class in order 

  max_num = max(class_count)                 # get the maximum of the quantity of images of all classes 

  for i in range(class_num):                 # copy and add data to one class if it is lack of data
    if class_count[i] == max_num:              # no change to the class with largerst quantity of images
      continue
    else:
      delta = max_num - class_count[i]         # randomly select images to copy and add to the class
      index_a = np.random.randint(0,class_count[i],delta)
      for _ in index_a:
        class_image[i].append(class_image[i][_])
  class_image = np.array(class_image)
  shape[0] = class_num * max_num
  class_image.resize(tuple(shape))    # remove the bracket between different classes

  class_label = np.array([class_label[i] for i in range(class_num) for _ in range(max_num)])  # expand the label array to correspond the expanded image array
  
  return class_image, class_label


"""
# basic data augmentation (horizontally flip, vertically flip, and diagonally flip)
# input: images
#        arrays
#        multi_depth
# output: augmented images and corresponding labels (both as Array)
"""
def augmentation(images, labels, multi_depth):

  print('augmentation')

  if type(images) is not np.ndarray:   # set the type of images to array
    images = np.array(images)
  if type(labels) is not np.ndarray:   # set the type of labels to array
    labels = np.array(labels)

  if multi_depth:                         # if image has multiple channels(depth)
    leftright_flip = images[:,:,:,::-1]      # horizontal flip
    updown_flip = images[:,:,::-1,:]         # vertical flip
    diag_flip = images[:,:,::-1,::-1]        # diagonal flip

  else:                                   # if image has only one channel
    leftright_flip = images[:,:,::-1]
    updown_flip = images[:,::-1,:]
    diag_flip = images[:,::-1,::-1]

  final_images = np.vstack((images,leftright_flip,updown_flip,diag_flip))

  if len(labels.shape) == 1:
    final_labels = np.hstack((labels,labels,labels,labels))  # original label
  else:
    final_labels = np.vstack((labels,labels,labels,labels))  # one_hot_code

  return final_images, final_labels



"""
# generate matrix for image matrix transformation
# input: size - the size of square matrix
# output: square matrix (each row or column has only one '1', others all set to '0')
"""
def generate_matrix(size):
  index = [i for i in range(size)]
  random.shuffle(index)
  index = np.array(index)
  index_offset = np.arange(size) * size
  matrix = np.zeros((size,size)) 
  matrix.flat[index_offset + index.ravel()] = 1
  return matrix

"""
# augment the images by randomly interchange rows and columns 
# input: images
#        labels
#        multi_depth
# output: augmented images and corresponding labels (both as Array)
#
"""
def extra_augmentation(images, 
                       labels, 
                       multi_depth, 
                       hchange = True, 
                       vchange = True):
  print('extra_augmentation')
  if type(images) is not np.ndarray:   # set the type of images to array
    images = np.array(images)
  if type(labels) is not np.ndarray:   # set the type of labels to array
    labels = np.array(labels)

  if multi_depth:
    size_m = images[0,0].shape[0]
    size_n = images[0,0].shape[1]
  else:
    size_m = images[0].shape[0]
    size_n = images[0].shape[1]

  temp = images
  if hchange:
    right_m = generate_matrix(size_n)
    temp = np.dot(temp, right_m)

  if vchange:
    left_m = generate_matrix(size_m)
    aug_images = []
    if multi_depth:
      for i in range(temp.shape[0]):
        temp_aug = []
        for j in range(temp.shape[1]):
          stack = np.dot(left_m, temp[i,j,:,:])
          temp_aug.append(stack)
        aug_images.append(temp_aug)
    else:
      for i in range(temp.shape[0]):
        stack = np.dot(left_m, temp[i,:,:])
        aug_images.append(stack)
    temp = aug_images

  final_images = np.vstack((images,temp))
  if len(labels.shape) == 1:
    final_labels = np.hstack((labels,labels))  # original label
  else:
    final_labels = np.vstack((labels,labels))  # one_hot_code

  return final_images, final_labels




def pretreat_data(data_type,
                  num_classes,
                  mode,
                  city_no,
                  skip_first_line = True,
                  seed=None):  
  """
  " input: data_type - choose which label to use. {0 : PM2.5, 1 : PM10, 2 : AQI}
  "        num_classes - for one_hot_code
  "        mode - set the mode to decide which type of images to read. { 0 : mod02_tif, 1 : mod02_rgb, 2 : mod04}
  "        skip_first_line - skip the first line or not when reading the labels csv files (default: True)
  "
  " output: train, validation and test datasets for each city {0: cd, 1: sn, 2: ms, 3:zy, 4: ls, 5: zg, 6: nj}
  "
  """  
  model_path = '/home/wu/ml_workspace/Multi_CNN_smog/model'  # where to read or write the model

  images, labels = read_datasets(mode, skip_first_line, test = False)      # read the images and labels
  test_images, test_labels = read_datasets(mode, skip_first_line, test = True) 

  multi_depth = (len(images.shape) == 4)

  city = {'cd':0, 'sn':1, 'ms':2, 'zy':3, 'ls':4, 'zg':5, 'nj':6}  # the index for each city
  x_y_cd = [0,160,0,40]        # the pixel range for each city
  x_y_sn = [160,240,0,40]
  x_y_ms = [0,60,40,100]
  x_y_zy = [60,240,40,100]
  x_y_ls = [0,40,100,180]
  x_y_zg = [40,150,100,180]
  x_y_nj = [150,240,100,180]
  x_y = np.array([x_y_cd, x_y_sn, x_y_ms, x_y_zy, x_y_ls, x_y_zg, x_y_nj])


  if multi_depth:
    images_temp = images[ :, :, x_y[city_no,2] : x_y[city_no,3], x_y[city_no,0] : x_y[city_no,1]]
    test_images = test_images[ :, :, x_y[city_no,2] : x_y[city_no,3], x_y[city_no,0] : x_y[city_no,1]]

  else:
    images_temp = images[ :, x_y[city_no,2] : x_y[city_no,3], x_y[city_no,0] : x_y[city_no,1]]
    test_images = test_images[ :, x_y[city_no,2] : x_y[city_no,3], x_y[city_no,0] : x_y[city_no,1]]
  
  labels_temp = labels[:,city_no,data_type]
  test_labels = test_labels[:,city_no,data_type]
  print(test_labels)

  images_temp, labels_temp = balance_label(images_temp, labels_temp) 

  images_temp, labels_temp = augmentation(images_temp, labels_temp, multi_depth)
  test_images, test_labels = augmentation(test_images, test_labels, multi_depth)
  for j in range(0):
#    images_temp, labels_temp = extra_augmentation(images_temp, labels_temp, multi_depth)
    test_images, test_labels = extra_augmentation(test_images, test_labels, multi_depth)

  labels_temp = dense_to_one_hot(labels_temp, num_classes)
  test_labels = dense_to_one_hot(test_labels, num_classes)

  num_images = images_temp.shape[0]
  index = [i for i in range(num_images)]
  random.shuffle(index)

  vali_end = int(0.2 * num_images)
#  train_end = int(0.6 * num_images)
  options = dict(seed=seed)

  train_image_temp = np.array(images_temp)
  train_label_temp = np.array(labels_temp)
  vali_image_temp  = np.array([images_temp[index[_]] for _ in range(vali_end)])
  vali_label_temp  = np.array([labels_temp[index[_]] for _ in range(vali_end)])
  test_images      = np.array(test_images)
  test_labels      = np.array(test_labels)

  train = DataSet(train_image_temp, train_label_temp, **options)
  vali  = DataSet(vali_image_temp,  vali_label_temp,  **options)
  test  = DataSet(test_images,      test_labels,      **options)
  data  = base.Datasets(train = train, validation = vali, test = test)

  return data


