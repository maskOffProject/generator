# requirements
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, add
from sklearn.model_selection import train_test_split
import random
from PIL import Image
import os
from maSequence import MaSequence

base_path = 'C:/mask'
masked_photos_path = base_path + '/with_mask/'
black_masked_photos_path = base_path + '/with_black_mask/'
unmasked_photos_path = base_path + '/without_mask/'

img_rows = 128
img_cols = 128
batch_size = 16


def process_image(path):
    full_image = Image.open(path).resize((img_rows, img_cols))
    rgb_img_array = (np.asarray(full_image)) / 255
    return rgb_img_array


def get_unmasked_photo_name_by_masked(masked_name):
    return masked_name.replace('with-mask-default-mask-', '')


def process_batch(masked_photos):
    X = []
    Y = []
    for photo in masked_photos:
        masked_photo_path = masked_photos_path + photo
        unmasked_photo_name = get_unmasked_photo_name_by_masked(photo)
        unmasked_photo_path = unmasked_photos_path + unmasked_photo_name
        if os.path.exists(masked_photo_path) and os.path.exists(unmasked_photo_path):
            masked_photo = process_image(masked_photo_path)
            X.append(masked_photo)
            unmasked_photo = process_image(unmasked_photo_path)
            Y.append(unmasked_photo)
            black_masked_photo_path = black_masked_photos_path + unmasked_photo_name
            if os.path.exists(black_masked_photo_path):
                black_masked_photo = process_image(black_masked_photo_path)
                X.append(black_masked_photo)
                Y.append(unmasked_photo)
    return X, Y


def print_masked_unmasked(masked, unmasked):
    plt.figure(figsize=(20, 5))
    masks = masked[:10]
    nomasks = unmasked[:10]
    for i, (x, y) in enumerate(zip(masks[:10], nomasks[:10])):
        plt.subplot(2, 10, i + 1)
        plt.imshow(x)
        plt.axis("OFF")

        plt.subplot(2, 10, i + 11)
        plt.imshow(y)
        plt.axis("OFF")
    plt.show()


def get_model(input_shape):
    # encoder
    In = Input(shape=input_shape)
    c1 = Conv2D(32, 3, activation="relu", padding="same")(In)
    c1 = Conv2D(32, 3, activation="relu", padding="same")(c1)
    m1 = MaxPooling2D(2)(c1)
    c2 = Conv2D(64, 3, activation="relu", padding="same")(m1)
    c2 = Conv2D(64, 3, activation="relu", padding="same")(c2)
    m2 = MaxPooling2D(2)(c2)
    c3 = Conv2D(128, 3, activation="relu", padding="same")(m2)
    c3 = Conv2D(128, 3, activation="relu", padding="same")(c3)
    m3 = MaxPooling2D(2)(c3)
    c4 = Conv2D(256, 3, activation="relu", padding="same")(m3)
    c4 = Conv2D(256, 3, activation="relu", padding="same")(c4)
    u1 = Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(c4)
    c5 = Conv2D(128, 3, activation="relu", padding="same")(u1)
    c5 = Conv2D(128, 3, activation="relu", padding="same")(c5)
    a1 = add([c5, c3])
    u2 = Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(a1)
    c6 = Conv2D(64, 3, activation="relu", padding="same")(u2)
    c6 = Conv2D(64, 3, activation="relu", padding="same")(c6)
    a2 = add([c6, c2])
    u3 = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(a2)
    c7 = Conv2D(32, 3, activation="relu", padding="same")(u3)
    c7 = Conv2D(32, 3, activation="relu", padding="same")(c7)
    a3 = add([c7, c1])
    Out = Conv2D(3, 3, activation="sigmoid", padding="same")(a3)

    model = Model(In, Out)
    # adam = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model


def show_loss(history):
    plt.figure(figsize=(20, 20))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.legend(["loss", "val_loss"])
    plt.show()


def show_one_prediction(masked, unmasked, model):
    plt.figure(figsize=(20, 5))
    masks = masked[:10]
    nomask_preds = model.predict(masks)
    nomask_actuals = unmasked[:10]
    for i in range(len(masks)):
        plt.subplot(3, 10, i + 1)
        plt.imshow(masks[i])
        plt.axis("OFF")

        plt.subplot(3, 10, i + 11)
        plt.imshow(nomask_preds[i])
        plt.axis("OFF")

        plt.subplot(3, 10, i + 21)
        plt.imshow(nomask_actuals[i])
        plt.axis("OFF")
    plt.show()


def show_test_results(test, model):
    plt.figure(figsize=(20, 5))

    X, Y = process_batch(test)

    nomask_preds = model.predict(np.array(X))
    for i in range(10):
        plt.subplot(3, 10, i + 1)
        plt.imshow(X[i])
        plt.axis("OFF")

        plt.subplot(3, 10, i + 11)
        plt.imshow(nomask_preds[i])
        plt.axis("OFF")

        plt.subplot(3, 10, i + 21)
        plt.imshow(Y[i])
        plt.axis("OFF")
    plt.show()


def main():
    masked_photos = os.listdir(masked_photos_path)[:1000]
    random.shuffle(masked_photos)
    train, test = np.split(masked_photos, [int(len(masked_photos) * 0.8)])
    X, Y = process_batch(train)
    X = np.array(X)
    Y = np.array(Y)
    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.1)
    print_masked_unmasked(x_val, y_val)
    gotrain = MaSequence(x_train, y_train, batch_size)
    goval = MaSequence(x_val, y_val, batch_size)

    model = get_model(x_train[0].shape)
    model.summary()

    checkpointer = ModelCheckpoint(filepath='generator.h5', verbose=0, save_best_only=True)

    history = model.fit(gotrain,
                        steps_per_epoch=gotrain.__len__(),
                        epochs=10,
                        callbacks=[checkpointer],
                        validation_data=goval,
                        validation_steps=goval.__len__())

    show_loss(history)

    model = keras.models.load_model("generator.h5")

    show_one_prediction(x_val, y_val, model)

    show_test_results(test, model)


if __name__ == "__main__":
    main()
