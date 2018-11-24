from keras.models import Sequential
from keras.layers import Conv2D, Activation, BatchNormalization, GlobalAvgPool2D

def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(16, (3, 3), strides=(2,2),padding="same", activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), strides=(2,2),padding="same", activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), strides=(2,2),padding="same", activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), strides=(2,2),padding="same", activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(num_classes, (3, 3), padding="same"))
    model.add(GlobalAvgPool2D())
    model.add(Activation("softmax"))
    return model
