import numpy as np  # linear algebra
import cv2

#load_data
images = np.load("/home/server/kaggle_dataset/readme/images.npy")
labels = np.load("/home/server/kaggle_dataset/readme/labels.npy")

#resize the original image(640x480) to (160,120) and convert to gray image
new_images = np.zeros((2000, 120, 160), dtype=np.float)
for i in range(images.shape[0]):
    res = cv2.resize(images[i], (160, 120), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    new_images[i] = gray
new_images = new_images.reshape((2000, 120, 160, 1))

from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Activation, Dense, Flatten, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from sklearn.model_selection import train_test_split

#define model
def MCNN_body_branch(input_flow, flow_mode='large'):
    if flow_mode == 'large':
        filter_num_initial, conv_len_initial, maxpooling_size = 16, 9, (2, 2)
    elif flow_mode == 'medium':
        filter_num_initial, conv_len_initial, maxpooling_size = 20, 7, (2, 2)
    elif flow_mode == 'small':
        filter_num_initial, conv_len_initial, maxpooling_size = 24, 5, (2, 2)
    else:
        print('Only small/medium/large modes.')
        return None
    x = Conv2D(filter_num_initial, (conv_len_initial, conv_len_initial), padding='same', activation='relu')(input_flow)
    x = MaxPooling2D(pool_size=maxpooling_size)(x)

    x = Conv2D(filter_num_initial * 2, (conv_len_initial - 2, conv_len_initial - 2), padding='same', activation='relu')(
        x)
    x = MaxPooling2D(pool_size=maxpooling_size)(x)
    x = Conv2D(filter_num_initial, (conv_len_initial - 2, conv_len_initial - 2), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=maxpooling_size)(x)

    x = Conv2D(filter_num_initial // 2, (conv_len_initial - 2, conv_len_initial - 2), padding='same',
               activation='relu')(x)
    x = MaxPooling2D(pool_size=maxpooling_size)(x)

    return x


def MCNN(weights=None, input_shape=(120, 160, 1)):
    input_flow = Input(shape=input_shape)
    branches = []
    for flow_mode in ['large', 'medium', 'small']:
        branches.append(MCNN_body_branch(input_flow, flow_mode=flow_mode))
    merged_feature_maps = Concatenate(axis=3)(branches)

    cnn_feature = Flatten()(merged_feature_maps)
    dense = Dense(64, activation='relu')(cnn_feature)
    count = Dense(1)(dense)
    model = Model(inputs=input_flow, outputs=count)
    return model


model = MCNN()
model.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['mae'])
X_train, X_test, y_train, y_test = train_test_split(new_images, labels, test_size=0.2, random_state=0)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.1, random_state=0)

model.fit(X_train, y_train, validation_data=(X_validation, y_validation),
          nb_epoch=200, batch_size=32, verbose=2)

score = model.evaluate(X_test, y_test, batch_size=32)
print(score)
