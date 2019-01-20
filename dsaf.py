# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

# Any results you write to the current directory are saved as output.

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import layers
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from keras.models import Sequential

def preprocessing_Image(data, m):
    print('Preprocessing images')
    X_train = np.zeros((m, 100, 100, 3))
    count = 0
    for fig in data['Image_path']:
        img = image.load_img(fig, target_size=(100,100,3))
        x = image.img_to_array(img)
        x = preprocess_input(x)
        X_train[count] = x
        if count%500 == 0:
            print('Processing Image: ',count+1, ",",fig)
        count += 1
    X_train /= 255  # 이미지 범위를 1 이하 범위로 한정
    return X_train

def preprocessing_Label(y):
    values = np.array(y)
    # Label 값을 숫자로 변경
    label_encoder =LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # Label 값을 One-hot Encoding 진행
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    y = onehot_encoded
    return y, label_encoder

def model(y):
    model = Sequential()
    # conv layer 1
    model.add(Conv2D(32, (7,7), strides=(1, 1), name='conv0', input_shape=(100, 100, 3)))
    model.add(BatchNormalization(axis = 3, name='bn0'))
    model.add(Activation('relu'))
    # Max pooling layer
    model.add(MaxPooling2D((2,2), name='max_pool'))
    # Conv Layer 2
    model.add(Conv2D(64, (3,3), strides=(1,1), name='conv1'))
    model.add(BatchNormalization(axis=3, name='bn1'))
    model.add(Activation('relu'))
    # Average Pooling layer
    model.add(AveragePooling2D((3,3), name='avg_pool'))
    # Dense Layer
    model.add(Flatten())
    model.add(Dense(y.shape[1]*2), name='r1')
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(y.shape[1], name='sm'))
    model.add(Activation('softmax'))
    # Loss Func, Optimizer
    optim = Adam(lr= 0.001, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorial_crossentropy', optimizer=optim, metrics=['accuracy'])
    model.summary()
    return model

def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]

def main():
    # input폴더 안의 파일들 확인
    print(os.listdir('../input'))
    # 이미지 폴더와 csv파일 변수로 저장
    img_train_path = os.path.abspath('../input/train')
    img_test_path = os.path.abspath('../input/test')
    csv_train_path = os.path.abspath('../input/train.csv')
    # img_train_path = os.path.abspath('E:/Data/Humpback Whale Identification/train')
    # img_test_path = os.path.abspath('E:/Data/Humpback Whale Identification/test/')
    # csv_train_path = os.path.abspath('E:/Data/Humpback Whale Identification/train.csv')

    train_df = pd.read_csv(csv_train_path)
    # Training 이미지 파일 이름 저장
    train_df['Image_path'] = [os.path.join(img_train_path, whale) for whale in train_df['Image']]
    y, label_encoder = preprocessing_Label(train_df['Id'])
    print(y.shape)
    X = preprocessing_Image(train_df, train_df.shape[0])
    X_train_cv, X_valid, y_train_cv, y_valid = train_test_split(X, y, random_state=1, train_size=0.7)

    # Test 이미지 파일 저장 --> Training 끝나고
    test = os.listdir(img_test_path)
    test_file_list = [os.path.join(img_test_path, whale) for whale in test]
    test_list = pd.DataFrame(test_file_list, columns=['Image_path'])
    col = ['Image']
    test_df = pd.DataFrame(test, columns=col)
    test_df['Id'] = ''
    X_test = preprocessing_Image(test_list, test_df.shape[0])

    file_path = ".model_weight.hdf5"
    callbacks = get_callbacks(file_path, patience=5)
    gmodel = model(y)

    while True:
            try:

                Input = int(input("You can select number. Training[1] / Test&Save[2] / Exit[3]"))
                if Input == 1:
                    history = gmodel.fit(X_train_cv, y_train_cv,
                               batch_size=12,
                               epochs=50,
                               verbose=1,
                               validation_data=(X_valid, y_valid),
                               callbacks=callbacks)
                    gmodel.load_weights(file_path)
                    score = gmodel.evaluate(X_valid, y_valid, verbose=1)
                    print('Test loss:', score[0])
                    print('Test accuracy:', score[1])

                    plt.plot(history.history['acc'])
                    plt.title('Model accuracy')
                    plt.ylabel('Accuracy')
                    plt.xlabel('Epoch')
                    plt.show()

                elif Input == 2:
                    gmodel.load_weights(file_path)
                    predicted_test = gmodel.predict(np.array(X_test), verbose=1)
                    for i, pred in enumerate(predicted_test):
                        # 각 결과마다 5개의 Label 선택
                        test_df.loc[i,'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))
                    test_df.to_csv('sub.csv', index=False)
                    print('Test Complete')
                elif Input == 3:
                    break
                else:
                    raise Exception('\n \t\t[1]Training,\t[2]Test&Save,\t[3]Exit\t')

            except NameError as err:
                print("1 OR 2 만 입력해주세요 ")

            except ValueError as err:
                print("only number")

            except KeyboardInterrupt:
                print("retry")


if __name__ == "__main__":
    main()