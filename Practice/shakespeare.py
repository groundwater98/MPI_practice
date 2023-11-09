import tensorflow as tf
import numpy as np
import math
from mpi4py import MPI
import random
import time
import tensorflow_datasets as tfds
from tensorflow.python.data.experimental import AUTOTUNE
from model import LSTM
import config as cf
comm = cf.comm
size = cf.size
rank = comm.Get_rank()

print("my rank is {}".format(rank))

path_to_file = tf.keras.utils.get_file('shakespeare.txt',
                                                    'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
seq_length = 100
batch_size = 64
buffer_size = int(10000 / size)
train_batch_size = int(batch_size/size)
vocab = sorted(set(text))
print("vocab's type: {}".format(type(vocab))) # list
print("vocab's length: {}".format(len(vocab)))
print("vocab: {}".format(vocab))

char2idx = {u: i for i, u in enumerate(vocab)}
print("char2idx's type: {}".format(type(char2idx))) #dict
print("char2idx's length: {}".format(len(char2idx)))
print("char2idx: {}".format(char2idx))

idx2char = np.array(vocab)
print("idx2char's type: {}".format(type(idx2char))) #numpy.ndarray
print("idx2char's length: {}".format(len(idx2char)))
print("idx2char: {}".format(idx2char))

text_as_int = np.array([char2idx[c] for c in text])
print("text_as_int's type: {}".format(type(text_as_int))) #numpy.ndarray
print("text_as_int's length: {}".format(len(text_as_int)))
print("text_as_int: {}".format(text_as_int))

# "text_as_int"를 기반으로 Dataset을 생성한다. 겹치지 않는 시퀀스를 추출하여 Dataset에 저장한다.
# text_as_int[i:i+seq_length+1]: 리스트 슬라이싱으로 i 부터 i+seq_length+1까지의 요소를 포함하는 부분 리스트를 반환
dataset = [text_as_int[i:i + seq_length + 1] for i in range(0, len(text_as_int), seq_length + 1)]
print("dataset's type: {}".format(type(dataset))) #numpy.ndarray
print("dataset's length: {}".format(len(dataset)))
print("dataset[0]: {}".format(dataset[0]))
print("dataset[0]'s length: {}".format(len(dataset[0])))

dataset_size = math.floor(len(dataset) / size)
print("dataset_size : {}".format(dataset_size))

val_batch_size = int(64/size)

# training batch 수 계산
num_train_batches = int(math.floor(dataset_size / (train_batch_size * size)))
num_val_batches = int(math.floor(buffer_size / (val_batch_size * size)))

#num_test_batches = buffer_size / test_batch_size
print("num_train_batches: {}".format(num_train_batches))
print("num_val_batches: {}".format(num_val_batches))
#print("num_test_batches: {}".format(num_test_batches))

# 각 process(또는 rank)에서 처리할 training sample 수를 계산한다.
num_local_train_samples = num_train_batches * train_batch_size
# 각 process(또는 rank)에서 처리할 validation sample 수를 계산한다.
num_local_valid_samples = num_val_batches * val_batch_size
# total validation sample 수를 계산한다.
num_valid_samples = num_val_batches * val_batch_size
#num_test_samples = test_batch_size * num_test_batches

print("num_local_train_samples: {}".format(num_local_train_samples))
print("num_local_valid_samples: {}".format(num_local_valid_samples))
print("num_val_samples: {}".format(num_valid_samples))
#print("num_test_samples: {}".format(num_test_samples))

# training dataset에서 현재 process(또는 rank)가 시작하는 위치를 나타냄
train_sample_offset = rank * num_local_train_samples
# validation dataset에서 현재 process(또는 rank)가 시작하는 위치를 나타냄
val_sample_offset = rank * num_local_valid_samples

#test_sample_offset = rank * dataset_size
print("train_sample_offset : {}".format(train_sample_offset))
print("val_sample_offset : {}".format(val_sample_offset))
#print("test_sample_offset : {}".format(test_sample_offset))


shuffled_index = np.arange(len(dataset), dtype='int32')
random.Random(time.time()).shuffle(shuffled_index)
comm.Bcast(shuffled_index, root=0)

print("shuffled_index's type: {}".format(type(shuffled_index)))
print("shuffled_index : {}".format(shuffled_index))


def read_train_data(self, sample_id):
    index = shuffled_index[self.train_sample_offset + sample_id.numpy()]
    print("index: {}".format(index))
    char = self.dataset[index]
    inp = char[:-1]
    target = char[1:]
    print("input: {}".format(inp))
    print("target: {}".format(target))
    return input, target


# Dataset은 tf.data.Dataset.from_tensor_slices 함수를 사용하여 np.arange(train_batch_size)를 기반으로 데이터셋을 생성한다.
dataset = tf.data.Dataset.from_tensor_slices(np.arange(train_batch_size))
# Dataset.map 함수를 사용하여 read_train_data 함수를 적용한다.
# tf.py_function을 사용하여 python함수를 Tensorflow 함수로 변환하고, num_parallel_calls 인자를 통해 병렬 처리 수준을 설정한다

dataset = dataset.map(lambda x: tf.py_function(read_train_data, inp=[x], Tout=[tf.int64, tf.int64]),
                      num_parallel_calls=AUTOTUNE)
# dataset.batch를 사용하여 데이터셋을 배치로 나눈다.
dataset = dataset.batch(train_batch_size)
# Dataset을 repeat()으로 반복시킴으로써 데이터셋이 끝나면 다시 처음으로 돌아가 반복시킨다.


dataset = dataset.repeat()
steps_per_epoch = int(num_local_train_samples / train_batch_size)
print("steps_per_epoch: {}".format(steps_per_epoch))
