##https://github.com/timctho/shufflenet-v2-tensorflow
##this code is in tf graph format,will have to change it to tf.keras form to use model.summary.
##We can go from keras to tf but not the other way around as tf graph is lower level than keras graph
#
#!pip install tensorflow-gpu=2.0.0
#import tensorflow as tf
#from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,SeparableConv2D,MaxPool2D,Dropout,AveragePooling2D,MaxPooling2D,Flatten,Dense,Input,Concatenate
#from tensorflow.keras.models import Model
#
#tf.__version__
##import tensorflow.keras as keras
##import tensorflow.contrib as tc
#
##slim = tc.slim
#
#def shuffle_unit(x, groups):
#    n, h, w, c = x.get_shape().as_list()
#    x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], h, w, groups, c // groups]))
#    x = tf.transpose(x, tf.convert_to_tensor([0, 1, 2, 4, 3]))
#    x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], h, w, c]))
#    return x
#
#
#def conv_bn_relu(x, out_channel, kernel_size, stride=1, dilation=1):
#    x = Conv2D(x, out_channel, kernel_size, stride, rate=dilation,
#                        biases_initializer=None, activation_fn=None)
#    x = BatchNormalization(x, activation_fn=tf.nn.relu, fused=False)
#    return x
#
#
#def conv_bn(x, out_channel, kernel_size, stride=1, dilation=1):
#    x = Conv2D(x, out_channel, kernel_size, stride, rate=dilation,
#                        biases_initializer=None, activation_fn=None)
#    x = BatchNormalization(x, activation_fn=None, fused=False)
#    return x
#
#
#def depthwise_conv_bn(x, kernel_size, stride=1, dilation=1):
#    x = SeparableConv2D(x, None, kernel_size, depth_multiplier=1, stride=stride,
#                                  rate=dilation, activation_fn=None, biases_initializer=None)
#    x = BatchNormalization(x, activation_fn=None, fused=False)
#    return x
#
#
#def resolve_shape(x):
#    n, h, w, c = x.get_shape().as_list()
#    if h is None or w is None:
#        kernel_size = tf.convert_to_tensor([tf.shape(x)[1], tf.shape(x)[2]])
#    else:
#        kernel_size = [h, w]
#    return kernel_size
#
#
#def global_avg_pool2D(x):
#    kernel_size = resolve_shape(x)
#    x = AveragePooling2D(x, kernel_size, stride=1)
#    x.set_shape([None, 1, 1, None])
#    return x
#
#
#def se_unit(x, bottleneck=2):
#    n, h, w, c = x.get_shape().as_list()
#    kernel_size = resolve_shape(x)
#    x_pool =AveragePooling2D(x, kernel_size, stride=1)
#    x_pool = tf.reshape(x_pool, shape=[-1, c])
#    fc =Dense(x_pool, bottleneck, activation_fn=tf.nn.relu,
#                                  biases_initializer=None)
#    fc = Dense(fc, c, activation_fn=tf.nn.sigmoid,
#                                  biases_initializer=None)
#    if n is None:
#        channel_w = tf.reshape(fc, shape=tf.convert_to_tensor([tf.shape(x)[0], 1, 1, c]))
#    else:
#        channel_w = tf.reshape(fc, shape=[n, 1, 1, c])
#        x = tf.multiply(x, channel_w)
#    return x
#
#
#def shufflenet_v2_block(x, out_channel, kernel_size, stride=1, dilation=1, shuffle_group=2):
#    if stride == 1:
#        top, bottom = tf.split(x, num_or_size_splits=2, axis=3)
#        half_channel = out_channel // 2
#        top = conv_bn_relu(top, half_channel, 1)
#        top = depthwise_conv_bn(top, kernel_size, stride, dilation)
#        top = conv_bn_relu(top, half_channel, 1)
#        out = tf.concat([top, bottom], axis=3)
#        out = shuffle_unit(out, shuffle_group)
#    else:
#        half_channel = out_channel // 2
#        b0 = conv_bn_relu(x, half_channel, 1)
#        b0 = depthwise_conv_bn(b0, kernel_size, stride, dilation)
#        b0 = conv_bn_relu(b0, half_channel, 1)
#        b1 = depthwise_conv_bn(x, kernel_size, stride, dilation)
#        b1 = conv_bn_relu(b1, half_channel, 1)
#        out = tf.concat([b0, b1], axis=3)
#        out = shuffle_unit(out, shuffle_group)
#        return out
#
############################Model##########################################
#        
#class ShuffleNetV2():
#
#    first_conv_channel = 24
#
#    def __init__(self, input_holder, cls, model_scale=1.0, shuffle_group=2, is_training=True):
#        tf.__version__
#        self.input = input_holder
#        self.output = None
#        self.cls = cls
#        self.shuffle_group = shuffle_group
#        self.channel_sizes = self._select_channel_size(model_scale)
#        self._build_model()
#
#    def _select_channel_size(self, model_scale):
#        # [(out_channel, repeat_times), (out_channel, repeat_times), ...]
#        if model_scale == 0.5:
#            return [(48, 4), (96, 8), (192, 4), (1024, 1)]
#        elif model_scale == 1.0:
#            return [(116, 4), (232, 8), (464, 4), (1024, 1)]
#        elif model_scale == 1.5:
#            return [(176, 4), (352, 8), (704, 4), (1024, 1)]
#        elif model_scale == 2.0:
#            return [(244, 4), (488, 8), (976, 4), (2048, 1)]
#        else:
#            raise ValueError('Unsupported model size.')
#
#    def _build_model(self):
#
#        out = conv_bn_relu(self.input, self.first_conv_channel, 3, 2)
#        out = MaxPool2D(out, 3, 2, padding='SAME')
#
#        for idx, block in enumerate(self.channel_sizes[:-1]):
#            out_channel, repeat = block
#            # First block is downsampling
#            out = shufflenet_v2_block(out, out_channel, 3, 2, shuffle_group=self.shuffle_group)
#            # Rest blocks
#            for i in range(repeat-1):
#                out = shufflenet_v2_block(out, out_channel, 3, shuffle_group=self.shuffle_group)
#            
#            out = conv_bn_relu(out, self.channel_sizes[-1][0], 1)
#
#            out = global_avg_pool2D(out)
#            out = Conv2D(out, self.cls, 1, activation_fn=None, biases_initializer=None)
#            out = tf.reshape(out, shape=[-1, self.cls])
#            out = tf.identity(out, name='cls_prediction')
#            #self.output = out
#            
#        model = Model(inputs=input, outputs=out)
#        return model
#    
#
##model=ShuffleNetV2()
##model._build_model
#



#https://github.com/timctho/shufflenet-v2-tensorflow
#this code is in tf graph format,will have to change it to tf.keras form to use model.summary.
#We can go from keras to tf but not the other way around as tf graph is lower level than keras graph

#!pip install tensorflow-gpu=2.0.0
#import tensorflow as tf
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,SeparableConv2D,MaxPool2D,Dropout,AveragePooling2D,MaxPooling2D,Flatten,Dense,Input,Concatenate
from tensorflow.keras.models import Model

tf.__version__
#import tensorflow.keras as keras
#import tensorflow.contrib as tc

#slim = tc.slim

def shuffle_unit(x, groups):
    n, h, w, c = x.get_shape().as_list()
    x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], h, w, groups, c // groups]))
    x = tf.transpose(x, tf.convert_to_tensor([0, 1, 2, 4, 3]))
    x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], h, w, c]))
    return x


def conv_bn_relu(x, out_channel, kernel_size, stride=1, dilation=1):
    x = Conv2D(out_channel, kernel_size, strides=stride, dilation_rate=dilation,
                        bias_initializer=None, activation=None)(x)
    x = BatchNormalization(fused=False)(x)
    x = Activation('relu')(x)
    return x


def conv_bn(x, out_channel, kernel_size, stride=1, dilation=1):
    x = Conv2D( out_channel, kernel_size, strides=stride, dilation_rate=dilation,
                        bias_initializer=None, activation=None)(x)
    x = BatchNormalization(fused=False)(x)
    
    return x


def depthwise_conv_bn(x, kernel_size, stride=1, dilation=1):
    x = SeparableConv2D(1, kernel_size, depth_multiplier=1, strides=stride,dilation_rate=dilation, activation=None, bias_initializer=None)(x) #Tapan
    x = BatchNormalization(fused=False)(x)
    return x


def resolve_shape(x):
    n, h, w, c = x.get_shape().as_list()
    if h is None or w is None:
        kernel_size = tf.convert_to_tensor([tf.shape(x)[1], tf.shape(x)[2]])
    else:
        kernel_size = [h, w]
    return kernel_size


def global_avg_pool2D(x):
    kernel_size = resolve_shape(x)
    x = AveragePooling2D( kernel_size, strides=1)(x)
    x.set_shape([None, 1, 1, None])
    return x


def se_unit(x, bottleneck=2):
    n, h, w, c = x.get_shape().as_list()
    kernel_size = resolve_shape(x)
    x_pool =AveragePooling2D( kernel_size, strides=1)(x)
    x_pool = tf.reshape(x_pool, shape=[-1, c])
    fc =Dense( bottleneck, activation=tf.nn.relu,bias_initializer=None)(x_pool)
    fc = Dense(c,bias_initializer=None)(fc)
    fc= Activation('softmax')(fc)
    if n is None:
        channel_w = tf.reshape(fc, shape=tf.convert_to_tensor([tf.shape(x)[0], 1, 1, c]))
    else:
        channel_w = tf.reshape(fc, shape=[n, 1, 1, c])
        x = tf.multiply(x, channel_w)
    return x


def shufflenet_v2_block(x, out_channel, kernel_size, stride=1, dilation=1, shuffle_group=2):
    if stride == 1:
        top, bottom = tf.split(x, num_or_size_splits=2, axis=3)
        half_channel = out_channel // 2
        top = conv_bn_relu(top, half_channel, 1)
        top = depthwise_conv_bn(top, kernel_size, stride, dilation)
        top = conv_bn_relu(top, half_channel, 1)
        out = tf.concat([top, bottom], axis=3)
        out = shuffle_unit(out, shuffle_group)
    else:
        half_channel = out_channel // 2
        b0 = conv_bn_relu(x, half_channel, 1)
        b0 = depthwise_conv_bn(b0, kernel_size, stride, dilation)
        b0 = conv_bn_relu(b0, half_channel, 1)
        b1 = depthwise_conv_bn(x, kernel_size, stride, dilation)
        b1 = conv_bn_relu(b1, half_channel, 1)
        out = tf.concat([b0, b1], axis=3)
        out = shuffle_unit(out, shuffle_group)
        return out

###########################Model##########################################
        
class ShuffleNetV2():

    first_conv_channel = 24

    def __init__(self, input_holder, cls, model_scale=1.0, shuffle_group=2, is_training=True):
        tf.__version__
        self.input = Input(shape=(input_holder[0], input_holder[1], input_holder[2],))
        self.output = None
        self.cls = cls
        self.shuffle_group = shuffle_group
        self.channel_sizes = self._select_channel_size(model_scale)
        self._build_model()

    def _select_channel_size(self, model_scale):
        # [(out_channel, repeat_times), (out_channel, repeat_times), ...]
        if model_scale == 0.5:
            return [(48, 4), (96, 8), (192, 4), (1024, 1)]
        elif model_scale == 1.0:
            return [(116, 4), (232, 8), (464, 4), (1024, 1)]
        elif model_scale == 1.5:
            return [(176, 4), (352, 8), (704, 4), (1024, 1)]
        elif model_scale == 2.0:
            return [(244, 4), (488, 8), (976, 4), (2048, 1)]
        else:
            raise ValueError('Unsupported model size.')

    def _build_model(self):

        out = conv_bn_relu(self.input, self.first_conv_channel, 3, 2)
        out = MaxPool2D( 3, 2, padding='SAME')(out)

        for idx, block in enumerate(self.channel_sizes[:-1]):
            out_channel, repeat = block
            # First block is downsampling
            out = shufflenet_v2_block(out, out_channel, kernel_size=3, stride=2, shuffle_group=self.shuffle_group)
            # Rest blocks
            for i in range(repeat-1):
                out = shufflenet_v2_block(out, out_channel, kernel_size=3, shuffle_group=self.shuffle_group)
            
            out = conv_bn_relu(out, self.channel_sizes[-1][0], 1)

            out = global_avg_pool2D(out)
            out = Conv2D( self.cls, 1, activation=None, bias_initializer=None)(out)
            out = tf.reshape(out, shape=[-1, self.cls])
            out = tf.identity(out, name='cls_prediction')
            #self.output = out
            

        model = Model(inputs=input, outputs=out)
        return model
    

#model=ShuffleNetV2()
#model._build_model

