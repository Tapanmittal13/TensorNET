import tensorflow as tf
import numpy as np  
import pandas as pd


#wrapper functions to make features of data (take a value & Wrap it inside of byteslist or int of size 64)
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def createDataRecord(out_filename,category,addrs,labels):
    #sample: createDataRecord('train.tfrecords',"Train data", X_train, Y_train)
    # open the TFRecords file
    writer = tf.compat.v1.python_io.TFRecordWriter(out_filename)   #changes done for 2.0
    for i in range(len(addrs)):
        # print how many images are saved every 1000 images
        if not i % 1000:
            print(category,': {}/{}'.format(i, len(addrs)))
            sys.stdout.flush()
        # Load the image
        img = addrs[i]
        label = labels[i]
        if img is None:
            print('Image not found')
            continue

        # Create a feature with the wrapping of bytes for the image and the wrap the label with an int 64
        feature = {
            'image_raw': _bytes_feature(tf.compat.as_bytes(img.tostring())),
            'label': _int64_feature(int(label))
        }
        # Create an example protocol buffer(sort of wrapping in an example)
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize example to string  and write on the file
        writer.write(example.SerializeToString())
        
    writer.close()
    sys.stdout.flush()

#take one record out from .tfrecords file so we get an image and a label
#augmentations will come inside this
def parser(record,img_height, img_width, channel,num_classes):
    #first say what kind of feature in the file i.e. 
    keys_to_features = {
        "image_raw": tf.compat.v2.io.FixedLenFeature([], tf.string),  #changes done for 2.0
        "label":     tf.compat.v2.io.FixedLenFeature([], tf.int64)
    }
    parsed = tf.io.parse_single_example(record, keys_to_features)  #read one record from the file   #changes done for 2.0
    image = tf.io.decode_raw(parsed["image_raw"], tf.uint8) #take image out of it and Convert the data from string back to the numbers: #changes done for 2.0
    image = tf.cast(image, tf.float32)  #convert it to float depending on the model need
    image = tf.reshape(image, shape=[img_height, img_width, channel])
    label = tf.cast(parsed["label"], tf.int32)  #Cast Label too
    label = tf.one_hot(label, num_classes)


    return image, label

def input_fn(filenames,buffer_size,seed,batch_size,GPU_buffer_size):
  
  
  dataset = tf.data.TFRecordDataset(filenames=filenames, buffer_size=buffer_size)# num_parallel_reads=40) #this tf.data API,one of the most imp in Tensorflow

  dataset=dataset.shuffle(buffer_size)
  dataset=dataset.repeat(seed)
  dataset=dataset.map(parser)
  dataset=dataset.batch(batch_size)

  dataset = dataset.apply(tf.data.experimental.prefetch_to_device('/GPU:0', buffer_size=GPU_buffer_size)) #changes done for 2.0
  dataset = dataset.prefetch(10)
  
  return dataset


class CreateTFRecord:
    """docstring for CreateTFRecord"""
    '''
    EXP.
    createDataRecord('train.tfrecords',"Train data", X_train, Y_train)
    '''
    def __init__(self, out_filename,category,addrs,labels):
        super().__init__()
        self.out_filename = out_filename
        self.category = category
        self.addrs=addrs
        self.labels=labels
        createDataRecord(self.out_filename, self.category,self.addrs,self.labels)
        
class Get_TFRecordDataset(object):
    def __init__(self, filenames,img_height, img_width, channel,num_classes,buffer_size,seed,batch_size,GPU_buffer_size):
        self.filenames = filenames
        self.img_height=img_height
        self.img_width=img_width
        self.channel=channel
        self.num_classes=num_classes
        self.buffer_size=buffer_size
        self.seed=seed
        self.batch_size=batch_size
        self.GPU_buffer_size = GPU_buffer_size
        parser(record,self.img_height, self.img_width, self.channel,self.num_classes)
        input_fn(self.filenames,self.buffer_size,self.seed,self.batch_size,self.GPU_buffer_size)