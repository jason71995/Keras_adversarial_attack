import keras
from keras.datasets import mnist
import numpy as np
from scipy import misc

from models.CNN_5 import build_model
from utils.graphs import build_training_functions, build_attack_function


train_batch_size = 128
test_batch_size = 64
num_classes = 10
epochs = 5
image_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train.reshape((-1,) + image_shape)
x_test = x_test.reshape((-1,) + image_shape)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical(y_test, num_classes)

num_of_train_data = x_train.shape[0]
num_of_test_data = x_test.shape[0]


model = build_model(image_shape, num_classes)
train_func, test_func = build_training_functions(image_shape, num_classes, model)
attack_func = build_attack_function(image_shape, num_classes, model)


print ("============================== Training ==============================")

for e in range(epochs):

    train_step_loss = []
    train_step_acc = []

    # random shuffle
    rand_idx = np.random.permutation(num_of_train_data)
    x_train = x_train[rand_idx]
    y_train = y_train[rand_idx]

    for i in range(0, num_of_train_data, train_batch_size):
        j = min(i + train_batch_size, num_of_train_data)
        batch_images = x_train[i:j]
        batch_labels = y_train[i:j]

        loss, acc = train_func([batch_images, batch_labels, 1])
        train_step_loss.append(loss)
        train_step_acc.append(acc)

        print ("\rEpoch:[{0}/{1}], Steps:[{2}/{3}] loss: {4:.4}, acc: {5:.4}".format(
            e+1, epochs, j, num_of_train_data, np.mean(train_step_loss), np.mean(train_step_acc)
        ), end='')

    print("")


print ("============================== Testing ==============================")

test_loss = []
test_acc = []
attack_loss = []
attack_acc = []

eps = 0.1 * np.ones(shape=(1,))

for i in range(0, num_of_test_data, test_batch_size):
    j = min(i + test_batch_size, num_of_test_data)
    batch_images = x_test[i:j]
    batch_labels = y_test[i:j]

    loss, acc = test_func([batch_images, batch_labels, 0])
    test_loss.append(loss)
    test_acc.append(acc)

    attack_images, = attack_func([batch_images, batch_labels, eps, 0])
    loss, acc = test_func([attack_images, batch_labels, 0])
    attack_loss.append(loss)
    attack_acc.append(acc)


print ("\rtest_loss: {0:.4}, test_acc: {1:.4}, attack_loss: {2:.4}, attack_acc: {3:.4}".format(
    np.mean(test_loss), np.mean(test_acc),np.mean(attack_loss), np.mean(attack_acc)
))


print ("============================== Saving Images ==============================")
images = (x_test[:100]*255).astype("uint8")
images = np.reshape(images,(10,10,28,28))
images = np.transpose(images,(0,2,1,3))
images = np.reshape(images,(280,280))
misc.imsave("normal_images.png",images)

attack_images, = attack_func([x_test[:100], y_test[:100], eps, 0])
images = (attack_images*255).astype("uint8")
images = np.reshape(images,(10,10,28,28))
images = np.transpose(images,(0,2,1,3))
images = np.reshape(images,(280,280))
misc.imsave("attack_images.png",images)


