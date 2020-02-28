import tensorflow as tf
import numpy as np
import math
import os
from keras.models import Model
from keras.layers import (
    Input, Activation, Add, BatchNormalization, LeakyReLU, AveragePooling1D, 
    concatenate, Lambda, UpSampling1D, Subtract, Reshape)
from keras.optimizers import Adam
from keras.layers.convolutional import Conv1D
from keras import backend as K
 

class FCN:
    def __init__(self, input_size = (36500,3), filter_channel=30, kernal_sizes = 55):
        self.input_size = input_size
        self.filter_channel = filter_channel
        self.kernal_size = kernal_sizes

    def build(self, pretrained_weights = None, trainable = True):
        layers_0 = Input(self.input_size)
        self.output = self.compute_output(layers_0, trainable)
        self.model = Model(inputs = layers_0, outputs = self.output)
        self.model.compile(loss = 'mse', optimizer='adam', metrics = ['accuracy'])
        
        #model.summary()

        if(pretrained_weights):
            self.model.load_weights(pretrained_weights)
        return self.model

    def compute_output(self, layers_0, trainable=True):
        filter_channel = self.filter_channel
        kernal_size = self.kernal_size
        output = Conv1D(filter_channel, kernal_size, padding = 'same', trainable=trainable, name='FCN_C1')(layers_0)
        output = BatchNormalization(trainable=trainable, name='FCN_BN1')(output)
        output = LeakyReLU()(output)

        output = Conv1D(filter_channel, kernal_size, padding = 'same', trainable=trainable, name='FCN_C2')(output)
        output = BatchNormalization(trainable=trainable, name='FCN_BN2')(output)
        output = LeakyReLU()(output)

        output = Conv1D(filter_channel, kernal_size, padding = 'same', trainable=trainable, name='FCN_C3')(output)
        output = BatchNormalization(trainable=trainable, name='FCN_BN3')(output)
        output = LeakyReLU()(output)

        output = Conv1D(filter_channel, kernal_size, padding = 'same', trainable=trainable, name='FCN_C4')(output)
        output = BatchNormalization(trainable=trainable, name='FCN_BN4')(output)
        output = LeakyReLU()(output)

        output = Conv1D(filter_channel, kernal_size, padding = 'same', trainable=trainable, name='FCN_C5')(output)
        output = BatchNormalization(trainable=trainable, name='FCN_BN5')(output)
        output = LeakyReLU()(output)

        output = Conv1D(filter_channel, kernal_size, padding = 'same', trainable=trainable, name='FCN_C6')(output)
        output = BatchNormalization(trainable=trainable, name='FCN_BN6')(output)
        output = LeakyReLU()(output)

        output = Conv1D(filter_channel, kernal_size, padding = 'same', trainable=trainable, name='FCN_C7')(output)
        output = BatchNormalization(trainable=trainable, name='FCN_BN7')(output)
        output = LeakyReLU()(output)

        output = Conv1D(1, kernal_size, padding = 'same', trainable=trainable, name='FCN_C8')(output)
        return Activation('tanh')(output)

class SDUN:
    def __init__(self, input_size = (36500,2), filter_channel=128, kernal_sizes = [2,3,3,3], dilation_rates = [1,2,6,18]):
        self.input_size = input_size
        self.filter_channel = filter_channel
        self.kernal_sizes = kernal_sizes
        self.dilation_rates = dilation_rates

    def build(self, pretrained_weights = None):
        layers_0 = Input(self.input_size)

        output = self.compute_output(layers_0)
        self.model = Model(inputs = layers_0, outputs = output)
        self.model.compile(loss = 'mse', optimizer='adam', metrics = ['accuracy'])
        
        #model.summary()

        if(pretrained_weights):
            self.model.load_weights(pretrained_weights)
        return self.model

    def compute_output(self, layers_0):
        layers_1 = self.dilation_4_layers_54_receptive_field_with_BN_Relu(layers_0, self.filter_channel, self.kernal_sizes, self.dilation_rates)
        layers_2 = self.dilation_4_layers_54_receptive_field_with_BN_Relu(layers_1, self.filter_channel, self.kernal_sizes, self.dilation_rates)
        layers_3 = self.dilation_4_layers_54_receptive_field_with_BN_Relu(layers_2, self.filter_channel, self.kernal_sizes, self.dilation_rates)
        layers_3_2 = Add()([layers_3, layers_2])
        layers_4 = self.dilation_4_layers_54_receptive_field_with_BN_Relu(layers_3_2, self.filter_channel, self.kernal_sizes, self.dilation_rates)
        layers_4_1 = Add()([layers_4, layers_1])
        layers_5 = self.dilation_4_layers_54_receptive_field(layers_4_1, 1, self.kernal_sizes, self.dilation_rates)        
        return Activation('tanh')(layers_5)

    def dilation_4_layers_54_receptive_field_with_BN_Relu(self, input, filter_channel, kernal_sizes, dilation_rates):
        conv1 = Conv1D(filter_channel, kernal_sizes[0], padding = 'same', dilation_rate = dilation_rates[0])(input)
        conv2 = Conv1D(filter_channel, kernal_sizes[1], padding = 'same', dilation_rate = dilation_rates[1])(conv1)
        conv3 = Conv1D(filter_channel, kernal_sizes[2], padding = 'same', dilation_rate = dilation_rates[2])(conv2)
        conv4 = Conv1D(filter_channel, kernal_sizes[3], padding = 'same', dilation_rate = dilation_rates[3])(conv3)
        BN = BatchNormalization()(conv4)
        return LeakyReLU()(BN)
    
    def dilation_4_layers_54_receptive_field(self, input, filter_channel, kernal_sizes, dilation_rates):
        conv1 = Conv1D(filter_channel, kernal_sizes[0], padding = 'same', dilation_rate = dilation_rates[0])(input)
        conv2 = Conv1D(filter_channel, kernal_sizes[1], padding = 'same', dilation_rate = dilation_rates[1])(conv1)
        conv3 = Conv1D(filter_channel, kernal_sizes[2], padding = 'same', dilation_rate = dilation_rates[2])(conv2)
        conv4 = Conv1D(filter_channel, kernal_sizes[3], padding = 'same', dilation_rate = dilation_rates[3])(conv3)
        return conv4


class Sinc_Conv_Layer():
    def __init__(self, input_size, N_filt, Filt_dim, fs, NAME):
        # Mel Initialization of the filterbanks
        low_freq_mel = 80
        high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, N_filt)  # Equally spaced in Mel scale
        f_cos = (700 * (10**(mel_points / 2595) - 1)) # Convert Mel to Hz
        b1=np.roll(f_cos,1).astype(np.float32)
        b2=np.roll(f_cos,-1).astype(np.float32)
        b1[0]=30
        b2[-1]=(fs/2)-100
                
        self.freq_scale=fs*1.0
        self.NAME = NAME
        band = tf.constant((b2-b1)/self.freq_scale)
        b1 = tf.constant(b1/self.freq_scale)
        self.filt_b1 = tf.get_variable(self.NAME+"_b1", initializer=b1)
        self.filt_band = tf.get_variable(self.NAME+"_band", initializer=band)
        self.N_filt=N_filt
        self.Filt_dim=Filt_dim
        self.fs=fs
        self.input_size = input_size

    def input_channel_slice(self, x, slice_index):
        return tf.expand_dims(x[:, :, slice_index],axis=-1)

    def compute_output(self, input_tensor, slice_index):
        '''
        input_tensor = Reshape((self.input_size[0]*self.input_size[1],-1))(input_tensor)
        filted = Lambda(self.sinc_conv)(input_tensor)
        return Reshape(self.input_size)(filted)
        '''
        sliced_tensor = Lambda(self.input_channel_slice, arguments={'slice_index':slice_index})(input_tensor)
        return Lambda(self.sinc_conv)(sliced_tensor)
    
    def sinc_conv(self, input_tensor):
        N = self.Filt_dim
        t_right = tf.linspace(1.0, (N-1)/2, int((N-1)/2)) / self.fs

        min_freq = 50.0
        min_band = 50.0
        
        filt_beg_freq=tf.abs(self.filt_b1)+min_freq/self.freq_scale
        filt_end_freq=filt_beg_freq+(tf.abs(self.filt_band)+min_band/self.freq_scale)

        n=tf.linspace(0.0, N, N)

        # Filter window (hamming)
        window = 0.54-0.46*tf.cos(2*math.pi*n/N)
        filter_tmp = []
        for i in range(self.N_filt):
                        
            low_pass1 = 2*filt_beg_freq[i] * self.sinc(t_right*(2*math.pi*filt_beg_freq[i]*self.freq_scale))
            low_pass2 = 2*filt_end_freq[i] * self.sinc(t_right*(2*math.pi*filt_end_freq[i]*self.freq_scale))
            band_pass=(low_pass2-low_pass1)

            band_pass=band_pass/tf.reduce_max(band_pass)
            filter_tmp.append(band_pass*window)

        filters = tf.reshape(tf.stack(filter_tmp, axis=1),(self.Filt_dim, 1, self.N_filt), name=self.NAME+"_filters")
        out=tf.nn.conv1d(input_tensor, filters, stride=1, padding='SAME')
        return out

    def sinc(self, x):
        atzero = tf.divide(tf.sin(x),1)
        atother = tf.divide(tf.sin(x),x)
        value = tf.where(tf.equal(x,0), atzero, atother )
        return tf.concat([tf.reverse(value, axis=[0]), tf.constant(1,dtype=tf.float32, shape=[1,]), value], axis=0)


class rSDFCN:
    def __init__(self, input_size = (36500,2), filter_channel=128, kernal_sizes = [2,3,3,3], dilation_rates = [1,2,6,18],
                 FCN_weight = 'filepath.h5'):
        K.clear_session()
        self.input_size = input_size
        self.filter_channel = filter_channel
        self.kernal_sizes = kernal_sizes
        self.enhancer = SDUN(input_size = (input_size[0], 2+80*input_size[1]), filter_channel=filter_channel)
        self.denoiser = FCN(input_size = input_size, filter_channel=64)
        self.FCN_weight = FCN_weight
        
        self.feature_extracters = []
        for i in range(input_size[1]):
            self.feature_extracters.append(Sinc_Conv_Layer(input_size, N_filt = 80, 
                Filt_dim = 251, fs = 16000, NAME = "SincNet_"+str(i)))

    def build(self, pretrained_weights = None):
        layers_0 = Input(self.input_size)
        self.smoothed = self.denoiser.compute_output(layers_0, trainable=False)
        first_layer_features = [self.feature_extracters[i].compute_output(layers_0, i) for i in range(self.input_size[1])]
        hybrid = concatenate(first_layer_features+[self.smoothed], axis=2)
        self.enhanced = self.enhancer.compute_output(hybrid)
        output = Add()([self.smoothed,self.enhanced])
        adam = Adam(lr=0.001)
        self.model = Model(inputs = layers_0, outputs = output)
        self.model.compile(loss = 'mse', optimizer = adam, metrics = [self.FCN_loss])
        self.model.load_weights(self.FCN_weight, by_name=True)
        #model.summary()
        if(pretrained_weights):
            self.model.load_weights(pretrained_weights)
        return self.model

    def input_slice(self, x, i):
        return tf.expand_dims(x[:,:,i],axis=2)
    
    def FCN_loss(self, y_true, y_pred):
        return K.mean(K.square(self.smoothed - y_true), axis=-1)