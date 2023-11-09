import tensorflow as tf
from transformers import TFAutoModel
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# if embedding_dim=256, each word will be represented as a 256-dimensional real-value vector. 
# embedding_dim is related to the complexity of the model. Higer dimensions allow the model to capture more fine-grained features of words, but it also increase the computational cost required for training the model/ 
class LSTM(tf.keras.Model):
	def __init__(self, vocab_size, embedding_dim=256, rnn_units=1024):
		super(LSTM, self).__init__()
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim) # 밖에서 config로 계산
		self.lstm = tf.keras.layers.LSTM(rnn_units,
					return_sequences=True,
					stateful=True,
					recurrent_initializer='glorot_uniform')
		self.dense = tf.keras.layers.Dense(vocab_size)

	def call(self, inputs):
		x = self.embedding(inputs)
		x = self.lstm(x)
		output = self.dense(x)
		return output

class GRU(tf.keras.Model):
	def __init__(self, vocab_size, embedding_dim=256, gru_units=1024):
		super(GRU, self).__init__()
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
		self.gru = tf.keras.layers.GRU(gru_units,
                                       return_sequences=True,
                                       stateful=True,
                                       recurrent_initializer='glorot_uniform')
		self.dense = tf.keras.layers.Dense(vocab_size)

	def call(self, inputs):
		x = self.embedding(inputs)
		x = self.gru(x)
		output = self.dense(x)
		return output

class Albert:
	def __init__(self, model_name='albert-base-v2'):
		self.model_name = model_name
		self.model = TFAutoModel.from_pretrained(self.model_name)

	def get_model(self):
		return self.model

	def __call__(self):
		return self.model

class ResNet:
    def __init__(self, num_layers=34):
        self.input_shape = (32, 32, 3) # CIFAR-10: 32x32
        self.num_classes = 10
        self.num_layers = num_layers
        self.build_model()

    def resnet_block(self, x, filters, kernel_size=3, stride=1, conv_shortcut=True):
        if conv_shortcut:
            shortcut = Conv2D(4 * filters, 1, strides=stride)(x)
        else:
            shortcut = x

        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters, 1, strides=stride)(x)

        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters, kernel_size, padding='SAME')(x)

        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(4 * filters, 1)(x)

        x = Add()([shortcut, x])
        return x

    def bottleneck_block(self, x, filters, kernel_size=3, stride=1, conv_shortcut=True):
        if conv_shortcut:
            shortcut = Conv2D(4 * filters, 1, strides=stride)(x)
        else:
            shortcut = x

        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters, 1, strides=stride)(x)

        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters, kernel_size, padding='SAME')(x)

        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(4 * filters, 1)(x)

        x = Add()([shortcut, x])
        return x

    def build_model(self):
        if self.num_layers == 34:
            num_blocks = [3, 4, 6, 3]
            block_fn = self.resnet_block
        elif self.num_layers == 50:
            num_blocks = [3, 4, 6, 3]
            block_fn = self.bottleneck_block
        else:
            raise ValueError("Only ResNet 34 and ResNet 50 architectures are supported.")

        inputs = Input(shape=self.input_shape)
        x = Conv2D(64, 7, strides=2, padding='SAME')(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        for _ in range(num_blocks[0]):
            x = block_fn(x, 64, conv_shortcut=(_ == 0))

        for _ in range(num_blocks[1]):
            x = block_fn(x, 128, stride=2 if _ == 0 else 1, conv_shortcut=(_ == 0))

        for _ in range(num_blocks[2]):
            x = block_fn(x, 256, stride=2 if _ == 0 else 1, conv_shortcut=(_ == 0))

        for _ in range(num_blocks[3]):
            x = block_fn(x, 512, stride=2 if _ == 0 else 1, conv_shortcut=(_ == 0))

        x = GlobalAveragePooling2D()(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        self.model = Model(inputs, outputs)

    def __call__(self, inputs):
        return self.model(inputs)