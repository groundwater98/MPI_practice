import math
import random
import os
import tensorflow as tf
import numpy as np
from mpi4py import MPI
import time
import config as cf
import tensorflow_datasets as tfds
from tensorflow.python.data.experimental import AUTOTUNE
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split

class Feeder_Shakespeare:
    def __init__(self):
        self.comm = cf.comm
        self.size = cf.size
        self.rank = self.comm.Get_rank()
        self.train_batch_size = cf.shakespeare_config["batch_size"]
        self.val_batch_size = cf.shakespeare_config["batch_size"]

        print("Number of train_batch_size: " + str(self.train_batch_size))
        print("Number of val_batch_size: " + str(self.val_batch_size))

        # buffer_size를 self.size로 나누는 이유는 전체 데이터셋을 여러프로세스로 나누기 위함이다
        self.buffer_size = cf.shakespeare_config["buffer_size"]

        self.path_to_file = tf.keras.utils.get_file('shakespeare.txt',
                                                    'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
        # open the path_to_file suing the open function, read it in 'rb' mode, decode it with encoding='utf-8' to read text data
        self.text = open(self.path_to_file, 'rb').read().decode(encoding='utf-8')
        self.seq_length = 100
        self.vocab = sorted(set(self.text))
        # Create a self.char2idx dictionary that maps the index for each character in self.vocab. for each character, the index of that character is sotred characters
        self.char2idx = {u: i for i, u in enumerate(self.vocab)}
        # Convert the self.vocab list into a Numpy array and stores it in self.idx2char.
        # this array is used to find characters for the index
        self.idx2char = np.array(self.vocab)
        # For each character in self.text, use the self.char2idx dictionary to map the character to an integer, convert it into a Numpy array, and store it in self.text_as_int.
        # Converts the source text data into an integer sequence.
        self.text_as_int = np.array([self.char2idx[c] for c in self.text])

       # self.dataset = [self.text_as_int[i:i+self.seq_length+1] for i in range(0, len(self.text_as_int),self.seq_length + 1)]
        self.dataset = [self.text_as_int[i:i + self.seq_length + 1]
                        if (i + self.seq_length + 1) < len(self.text_as_int)
                        else np.concatenate(
            [self.text_as_int[i:], np.zeros(self.seq_length + 1 - (len(self.text_as_int) - i))])
                        for i in range(0, len(self.text_as_int), self.seq_length + 1)]

        self.dataset_size = math.floor(len(self.dataset)/self.size)

        self.num_train_batches = int(math.floor(self.dataset_size / (self.train_batch_size * self.size)))
        self.num_val_batches = int(math.floor(self.buffer_size / (self.val_batch_size * self.size)))

        self.num_local_train_samples = self.num_train_batches * self.train_batch_size
        self.num_local_valid_samples = self.num_val_batches * self.val_batch_size
        self.num_valid_samples = self.num_val_batches * self.val_batch_size

        print("Number of training batches: " + str(self.num_train_batches))
        print("Number of validation batches: " + str(self.num_val_batches))
        print("Number of local training samples: " + str(self.num_local_train_samples))
        print("Number of validation samples: " + str(self.num_valid_samples))

        self.train_sample_offset = self.rank * self.num_local_train_samples
        self.val_sample_offset = self.rank * self.num_local_valid_samples
        self.shuffle()

    def shuffle(self):
        self.shuffled_index = np.arange(len(self.dataset), dtype='int32')
        random.Random(time.time()).shuffle(self.shuffled_index)
        self.comm.Bcast(self.shuffled_index, root=0)

    def read_train_data(self, sample_id):
        # train_sample_offset은 각 프로세스에서 처리할 training dataset의 시작 위치
        # sample_id는 training dataset 내의 sample index를 나타내는 텐서
        index = self.shuffled_index[self.train_sample_offset + sample_id.numpy()]
        char = self.dataset[index]
        # Model에 입력으로 제공되는 시퀀스이고 이를 기반으로 다음 문자를 예측해야함
        inp = char[:-1]
        # Model이 예측해야 할 다음 문자를 나타낸다.
        target = char[1:]
        return inp, target

    def read_val_data(self, sample_id):
        index = self.shuffled_index[self.val_sample_offset + sample_id.numpy()]
        char = self.dataset[index]
        inp = char[:-1]
        target = char[1:]
        return inp, target


    def train(self):
        # 0부터 self.num_local_train_samples - 1까지의 정수 배열을 생성
        dataset = tf.data.Dataset.from_tensor_slices(np.arange(self.num_local_train_samples))
        # 생성된 데이터셋의 각 원소에 대해 self.read_train_data 함수를 적용한다.
        # tf.py_function을 사용하여 이 함수를 TensorFlow 연산으로 감싼다.
        dataset = dataset.map(lambda x: tf.py_function(self.read_train_data, inp = [x], Tout = [tf.int64,tf.int64]),num_parallel_calls=AUTOTUNE)
        # 데이터셋을 여러 배치로 분할한다. 각 배치의 크기는 self.train_batch_size로 결정한다.
        dataset = dataset.padded_batch(self.train_batch_size, padded_shapes=([None], [None]))
        dataset = dataset.repeat()
        # per epoch당 수행되는 steps(전체 Dataset을 한 번 모두 사용한 횟수)
        steps_per_epoch = int(self.num_local_train_samples / self.train_batch_size)
        print("steps_per_epoch: {}".format(steps_per_epoch))
        return dataset, steps_per_epoch

    def validation(self):
        dataset = tf.data.Dataset.from_tensor_slices(np.arange(self.num_valid_samples))
        dataset = dataset.map(lambda x: tf.py_function(self.read_val_data, inp=[x], Tout=[tf.int64, tf.int64]),
                              num_parallel_calls=AUTOTUNE)
        dataset = dataset.padded_batch(self.val_batch_size, padded_shapes=([None], [None]))

        dataset = dataset.repeat()
        return dataset

class Feeder_CIFAR10:
    def __init__(self):
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = cifar10.load_data()
        self.preprocess_data()
        self.split_data()
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.batch_size = 64

    def preprocess_data(self):
        self.train_images, self.test_images = self.train_images / 255.0, self.test_images / 255.0

    def split_data(self):
        self.train_images, self.valid_images, self.train_labels, self.valid_labels = train_test_split(self.train_images, self.train_labels, test_size=0.1, random_state=42)

    def distribute_data(self, data, labels):
        local_data = np.array_split(data, self.size)[self.rank]
        local_labels = np.array_split(labels, self.size)[self.rank]
        dataset = tf.data.Dataset.from_tensor_slices((local_data, local_labels))
        dataset = dataset.shuffle(buffer_size=10000).batch(self.batch_size)
        return dataset

    def train(self):
        train_dataset = self.distribute_data(self.train_images, self.train_labels)
        steps_per_epoch = len(self.train_images) // self.size // self.batch_size
        print("steps_per_epoch: {}".format(steps_per_epoch))
        return train_dataset, steps_per_epoch

    def validation(self):
        return self.distribute_data(self.valid_images, self.valid_labels)

    def test(self):
        return self.distribute_data(self.test_images, self.test_labels)

