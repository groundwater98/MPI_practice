import tensorflow as tf
from mpi4py import MPI
import numpy
import time
class Training_Shakespeare:
    def __init__(self, feeder, model):
        self.feeder = feeder
        self.model = model
        self.optimizer = tf.keras.optimizers.SGD()
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x)
            loss_value = self.loss_fn(y, logits)

        gradients = tape.gradient(loss_value, self.model.trainable_variables)
        print(gradients[0])
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss_value, logits

    @tf.function
    def evaluate_step(self, x, y):
        logits = self.model(x)
        loss_value = self.loss_fn(y, logits)
        return loss_value, logits

    def train(self, num_epochs):
        train_dataset, steps_per_epoch = self.feeder.train()
        val_dataset = self.feeder.validation()
        test_dataset = self.feeder.test()

        total_start_time = time.time()

        for epoch in range(num_epochs):
            start_time = time.time()
            epoch_loss = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
            # x: batch, y: label
            for step, (x, y) in enumerate(train_dataset):
                loss_value, logits = self.train_step(x, y)
                epoch_loss(loss_value)
                epoch_accuracy(y, logits)
                if step >= steps_per_epoch:
                    break

            train_loss = epoch_loss.result()
            train_accuracy = epoch_accuracy.result() * 100.0

            val_loss, val_accuracy = self.evaluate(val_dataset)
            elapsed_time = time.time() - start_time

            print("EPoch {}/{} - Elapsed Time: {:.2f} seconds".format(epoch + 1, num_epochs, elapsed_time))
            print("Epoch {}/{}".format(epoch + 1, num_epochs))
            print("Train Loss: {:.2f}".format(train_loss))
            print("Train Accuracy: {:.2f}%".format(train_accuracy))
            print("Validation Loss: {:.2f}".format(val_loss))
            print("Validation Accuracy: {:.2f}%".format(val_accuracy))
            print()

        test_loss, test_accuracy = self.evaluate(test_dataset)
        total_end_time = time.time()
        total_elapsed_time = total_end_time-total_start_time

        print("Total Training Time: {:.2f} seconds".format(total_elapsed_time))
        print("Test Loss: {:.2f}".format(test_loss))
        print("Test Accuracy: {:.2f}%".format(test_accuracy))

    def evaluate(self, dataset):
        loss_metric = tf.keras.metrics.Mean()
        accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()

        for x, y in dataset:
            loss_value, logits = self.evaluate_step(x, y)

            loss_metric(loss_value)
            accuracy_metric(y, logits)

        loss = loss_metric.result()
        accuracy = accuracy_metric.result() * 100.0
        return loss, accuracy
