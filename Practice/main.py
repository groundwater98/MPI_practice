import tensorflow as tf
from model import LSTM, GRU, ResNet
from feeder_mpi import Feeder_Shakespeare, Feeder_CIFAR10
from training_mpi import Training_Shakespeare, Training_CIFAR10
import config

feeder = Feeder_CIFAR10()
print("=====Feeder is ready=====")

#model = LSTM(vocab_size=len(feeder.vocab))
#model = GRU(vocab_size=len(feeder.vocab))
model = ResNet()
print("=====Model is ready=====")

trainer = Training_CIFAR10(feeder=feeder, model=model)
print("=====Trainer is ready.======")

trainer.train(epochs=10)

print("=====End=====")
print('\n')

