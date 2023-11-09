import tensorflow as tf
import numpy as np


class Feeder_Shakespeare:
    def __init__(self):
        self.path_to_file = tf.keras.utils.get_file('shakespeare.txt',
                                                    'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
        # open the path_to_file suing the open function, read it in 'rb' mode, decode it with encoding='utf-8' to read text data
        self.text = open(self.path_to_file, 'rb').read().decode(encoding='utf-8')
        self.seq_length = 100
        self.batch_size = 32
        # buffer_size represent the size of the buffer used to mix data.
        self.buffer_size = 10000
        self.vocab = sorted(set(self.text))

        # Create a self.char2idx dictionary that maps the index for each character in self.vocab. for each character, the in dex of that character is sotred characters
        self.char2idx = {u: i for i, u in enumerate(self.vocab)}

        # Convert the self.vocab list into a Numpy array and stores it in self.idx2char.
        # this array is used to find characters for the index
        self.idx2char = np.array(self.vocab)

        # For each character in self.text, use the self.char2idx dictionary to map the character to an integer, convert it into a Numpy array, and store it in self.text_as_int.
        # Converts the source text data into an integer sequence.
        self.text_as_int = np.array([self.char2idx[c] for c in self.text])
        # Convert self.text_as_int to tensor to create the TensorFlow dataset, self.char_dataset.
        # This dataset contains each integer as a separate element.
        self.char_dataset = tf.data.Dataset.from_tensor_slices(self.text_as_int)

        # Create self.sequences using self.char_dataset by batch in a sequence of seq_length + 1 size.
        # Each sequence consists of an input sequence and a corresponding label.
        self.sequences = self.char_dataset.batch(self.seq_length + 1, drop_remainder=True)

        # This creates the final dataset, self.dataset, consisting of inputs and labels.
        self.dataset = self.sequences.map(self.split_input_target)

    def split_input_target(self, chunk):
        input_text = chunk[:-1] #except the last element
        target_text = chunk[1:]
        return input_text, target_text

    def get_dataset(self, split):
        if split == 'train':
            dataset = self.dataset.shuffle(self.buffer_size).batch(self.batch_size, drop_remainder=True).repeat().prefetch(tf.data.experimental.AUTOTUNE)
        elif split == 'validation':
            dataset = self.dataset.take(self.buffer_size).batch(self.batch_size, drop_remainder=True).prefetch(
                tf.data.experimental.AUTOTUNE)
        else:  # split == 'test'
            dataset = self.dataset.skip(self.buffer_size).batch(self.batch_size, drop_remainder=True).prefetch(
                tf.data.experimental.AUTOTUNE)
        return dataset

    def train(self):
        train_dataset = self.get_dataset('train')
        steps_per_epoch = len(self.text) // (self.seq_length * self.batch_size)
        return train_dataset, steps_per_epoch

    def validation(self):
        val_dataset = self.get_dataset('validation')
        return val_dataset

    def test(self):
        test_dataset = self.get_dataset('test')
        return test_dataset


