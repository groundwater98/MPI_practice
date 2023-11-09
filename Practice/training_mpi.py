import tensorflow as tf
import numpy as np
from mpi4py import MPI
import time
import config as cf
# mixed precision은 float32, float16을 혼합하여 사용하는 방법이다.
# 기본적으로 딥러닝 모델은 모든 계산을 float32에서 수행한다. 그러나 최근의 GPU는 float 16에서도 훌륭한 성능을 발휘하며, 이를 이용하면 GPU 메모리 사용량을 줄일 수 있다.
# 가중치와 작은 텐서는 여전히 float32에서 유지되지만, 큰 텐서(activation function의 출력)은 float16에서 유지된다.
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# mixed precision이 활성화되도록 한다.
mixed_precision.set_global_policy('mixed_float16')

# Tensorflow의 실행 모드를 즉시 실행(eager execution)모드로 설정하는 함수다. TensorFlow에서 기본적으로 사용되는 실행 모드는 Graph mode이다.
# Graph 모드는 계산 그래피를 빌드하고 세션을 실행해야 결과를 확인할 수 있지만 즉시 실행 모드에서는 계산 그래프를 빌드하지 않고, 각각의 연산이 호출될 때마다 즉시 실행되어 결과를 반환한다.
tf.config.run_functions_eagerly(run_eagerly=True)

# 디버그 모드는 데이터셋 연산의 동작을 검사하고 문제를 식별하는 데 도움이 되는 추가적인 검사와 경고를 활성화한다.
# 예를 들어, 데이터셋에서 잘못된 크기의 배치가 생성되거나 데이터가 손실되는 경우에 발생한다.
# 디버그 모드를 사용하면 데이터셋 파이프라인에서 발생하는 문제를 식별하고 수정하는 데 도움이 된다.
tf.data.experimental.enable_debug_mode()

import time

class Training_CIFAR10:
    def __init__(self, model, feeder, learning_rate=0.001):
        self.model = model
        self.feeder = feeder
        self.optimizer = Adam(learning_rate=learning_rate)
        self.loss_fn = SparseCategoricalCrossentropy()
        self.comm = MPI.COMM_WORLD

    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(images)
            loss = self.loss_fn(labels, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        avg_gradients = [self.comm.allreduce(g, op=MPI.SUM) / self.comm.size for g in gradients]
        self.optimizer.apply_gradients(zip(avg_gradients, self.model.trainable_variables))
        return loss

    def train(self, epochs):
        train_dataset, steps_per_epoch = self.feeder.train()
        val_dataset = self.feeder.validation()
        total_start_time = time.time()
        for epoch in range(epochs):
            start_time = time.time()
            for step, (images, labels) in enumerate(train_dataset):
                loss = self.train_step(images, labels)
                print(f"Epoch {epoch} | Step {step} | Loss: {loss}")
                if step >= steps_per_epoch:
                    break

            val_loss = self.evaluate(val_dataset)
            print(f"Epoch {epoch} | Validation Loss: {val_loss}")
            print(f"Time taken for epoch: {time.time() - start_time} secs\n")

        print(f"Total training time: {time.time() - total_start_time} secs")

    def evaluate(self, dataset):
        total_loss = 0
        num_batches = 0
        for images, labels in dataset:
            predictions = self.model(images)
            loss = self.loss_fn(labels, predictions)
            total_loss += loss
            num_batches += 1

        return total_loss / num_batches


class Training_Shakespeare:
    def __init__(self, feeder, model):
        self.feeder = feeder
        self.model = model
        self.optimizer = tf.keras.optimizers.SGD()
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.comm = cf.comm
        print("=====MPI communicator is ready=====")
        self.rank = self.comm.Get_rank()
        print("my rank is {}".format(self.rank))
        self.size = self.comm.Get_size()
        print("get size: {}".format(self.size))

    #@tf.function
    # x는 input data, y는 label
    def train_step(self, x, y):
        # tf.GradientTape()는 Tensorflow에서 제공하는 자동 미분 도구이고 이 구문안에서 발생한 모든 연산을 기록한다.
        with tf.GradientTape() as tape:
            logits = self.model(x)
            loss_value = self.loss_fn(y, logits)

        gradients = tape.gradient(loss_value, self.model.trainable_variables)
        indexed_slices_grad = gradients[0]
        dense_grads = gradients[1:]
        # dense_grads 리스트에 있는 각 gradients를 numpy 배열로 변환하여 grad_list에 저장한다.
        # grad_list는 5개
        grad_list = [tf.make_ndarray(tf.make_tensor_proto(g)) for g in dense_grads]
        # average the gradients across all processes
        averaged_gradients = [np.array(self.comm.allreduce(g, op=MPI.SUM)) / self.size for g in grad_list]
        # convert averaged gradients back into tf.Tensors
        averaged_gradients = [tf.convert_to_tensor(avg_g, dtype=g.dtype) for avg_g, g in
                              zip(averaged_gradients, dense_grads)]
        averaged_gradients = [indexed_slices_grad] + averaged_gradients
        # model parameter은 6개
        # 최종 gradients와 model의 weights를 묶어서 optimizer에 적용하여 model의 weights를 update한다.
        self.optimizer.apply_gradients(zip(averaged_gradients, self.model.trainable_variables))
        return loss_value, logits


    @tf.function
    def evaluate_step(self, x, y):
        logits = self.model(x)
        loss_value = self.loss_fn(y, logits)
        return loss_value, logits

    def train(self, num_epochs):
        train_dataset, steps_per_epoch = self.feeder.train()
        val_dataset = self.feeder.validation()
        total_start_time = time.time()
        cnt = 0
        print("=====Datasets are ready=====")

        for epoch in range(num_epochs):
            start_time = time.time()
            epoch_loss = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
            # x: batch, y: label
            for step, (x, y) in enumerate(train_dataset):
                print("step {}".format(cnt))
                cnt += 1
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

        total_end_time = time.time()
        total_elapsed_time = total_end_time - total_start_time

        print("Total Training Time: {:.2f} seconds".format(total_elapsed_time))

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
