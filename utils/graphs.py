from keras.optimizers import Adam
import keras.backend as K


def build_training_functions(input_shape, num_classes, model):

    images = K.placeholder((None,) + input_shape)
    y_true = K.placeholder((None, num_classes))
    y_pred = model(images)

    loss = K.mean(K.categorical_crossentropy(y_true, y_pred))
    acc  = K.mean(K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx()))

    # get updates of untrainable updates.
    # e.g. mean and variance in BatchNormalization
    untrainable_updates = model.get_updates_for([images])

    # get updates of trainable updates.
    trainable_updates = Adam(lr=0.0001).get_updates(loss, model.trainable_weights)

    # K.learning_phase() is required if model has different behavior during train and test.
    # e.g. BatchNormalization, Dropout
    train_func = K.function([images, y_true, K.learning_phase()], [loss, acc], untrainable_updates + trainable_updates)
    test_func  = K.function([images, y_true, K.learning_phase()], [loss, acc])

    return train_func, test_func


def build_attack_function(input_shape, num_classes, model):

    images = K.placeholder((None,) + input_shape)
    y_true = K.placeholder((None, num_classes))
    y_pred = model(images)

    loss = K.mean(K.categorical_crossentropy(y_true, y_pred))

    eps = K.placeholder((1,))
    grad = K.gradients(loss, [images])[0]
    r_adv = eps * K.sign(grad)
    attack_images = K.clip(images + r_adv, 0.0, 1.0)

    attack_function = K.function([images, y_true, eps, K.learning_phase()],[attack_images])
    return attack_function