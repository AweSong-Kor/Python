# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 19:38:21 2016

@author: HyunMin-Kor
"""

import numpy as np
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

class DataSet(object):
    def __init__(self, data, labels, dtype=dtypes.float32):
        dtype = dtypes.as_dtype(dtype).base_dtype
        
        self._num_examples = data.shape[0]
        self._data = data
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
    
    @property
    def data(self):
        return self._data
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._data = self._data[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        return self._data[start:end], self._labels[start:end]

def read_data_sets(data_dir, dtype=dtypes.float32, validation_size=2000):
    TRAIN_DATA = np.load(data_dir+'dos_train_data'+'.npy')
    TRAIN_LABELS = np.load(data_dir+'dos_train_label'+'.npy')
    TEST_DATA = np.load(data_dir+'dos_test_data'+'.npy')
    TEST_LABELS = np.load(data_dir+'dos_test_label'+'.npy')
    
    TRAIN_LABELS.astype('float32')
    TEST_LABELS.astype('float32')    
    
    validation_data = TRAIN_DATA[:validation_size]
    validation_labels = TRAIN_LABELS[:validation_size]
    train_data = TRAIN_DATA[validation_size:]
    train_labels = TRAIN_LABELS[validation_size:]
    
    train = DataSet(train_data, train_labels, dtype=dtype)
    validation = DataSet(validation_data, validation_labels, dtype=dtype)
    test = DataSet(TEST_DATA,TEST_LABELS,dtype=dtype)
    
    return base.Datasets(train=train, validation=validation, test=test)

def load_dos(data_dir='/home/gnos/work/data/'):
    return read_data_sets(data_dir)
    