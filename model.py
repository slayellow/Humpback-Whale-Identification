# 2019/01/21
# 수정해야할 부분
# Data Imbalance --> Kernel 보고 수정
# Model --> Image Classification 모델 구현해서 정리
# 코드 간결화

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from imageclassifier.Resnet import Resnet


def prepareImages(data, m, dataset):
    X_train = np.zeros((m, 224, 224, 3))
    count = 0
    if dataset == 'test':
        img = image.load_img("E:/Data/Humpback Whale Identification/" + dataset + "/" + data, target_size=(224, 224, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)
        X_train[count] = x
        count += 1
        return X_train
    else:
        for fig in data:
            # load images into images of size 100x100x3
            img = image.load_img("E:/Data/Humpback Whale Identification/" + dataset + "/" + fig, target_size=(224, 224, 3))
            x = image.img_to_array(img)
            x = preprocess_input(x)
            X_train[count] = x
            count += 1
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

def get_callbacks(filepath, patience=2):
    es = EarlyStopping('loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, monitor='acc', save_best_only=True)
    return [es, msave]

def batch_generator(X, y, batch_size=16):
    print('generator initiated')
    count = 0
    while True:
        # choose batch_size random images / labels from the data
        idx = np.random.randint(0, X.shape[0], batch_size)
        X_t = X.loc[idx,'Image']
        X_train = prepareImages(X_t, X_t.shape[0], 'train')
        X_train /= 255
        X_train = np.array(X_train)
        label = y[idx]
        yield X_train, label
        count += 1

def main():
    # input폴더 안의 파일들 확인
    # print(os.listdir('../input'))
    # 이미지 폴더와 csv파일 변수로 저장
    # img_train_path = os.path.abspath('../input/train')
    # img_test_path = os.path.abspath('../input/test')
    # csv_train_path = os.path.abspath('../input/train.csv')
    img_train_path = os.path.abspath('E:/Data/Humpback Whale Identification/train')
    img_test_path = os.path.abspath('E:/Data/Humpback Whale Identification/test')
    csv_train_path = os.path.abspath('E:/Data/Humpback Whale Identification/train.csv')

    train_df = pd.read_csv(csv_train_path)
    train_df['Image_path'] = [os.path.join(img_train_path, whale) for whale in train_df['Image']]
    y, label_encoder = preprocessing_Label(train_df['Id'])

    train_gen = batch_generator(train_df, y, batch_size=12)


    file_path = ".model_weight.hdf5"
    callbacks = get_callbacks(file_path, patience=5)
    #model = Alexnet(input_shape=(224,224,3), num_classes=y.shape[1])
    #model = VGG('E', input_shape=(224,224,3), num_classes=y.shape[1])
    model = Resnet(input_shape=(224,224,3), num_classes=y.shape[1])
    gmodel = model.model


    while True:
            try:

                Input = int(input("You can select number. Training[1] / Test&Save[2] / Exit[3]"))
                if Input == 1:
                    history = gmodel.fit_generator(generator=train_gen,
                           epochs=2,
                           steps_per_epoch= train_df.shape[0] // 12,
                                                   verbose=1,
                           callbacks=callbacks)
                    '''
                    self.model.fit_generator(
                        generator        = train_batch,
                        validation_data  = valid_batch,
                        validation_steps = validation_steps,
                        steps_per_epoch  = steps_per_epoch,
                        epochs           = epochs,
                        callbacks        = [ES],
                        verbose          = 1,
                        workers          = 3,
                        max_queue_size   = 8)
                  '''
                    gmodel.load_weights(file_path)

                    plt.plot(history.history['acc'])
                    plt.title('Model accuracy')
                    plt.ylabel('Accuracy')
                    plt.xlabel('Epoch')
                    plt.show()

                elif Input == 2:
                    # Test 과정 --> 수정해야할 부분
                    predicted_test = []
                    gmodel.load_weights(file_path)
                    test = os.listdir(img_test_path)
                    col = ['Image']
                    test_df = pd.DataFrame(test, columns=col)
                    test_df['Id'] = ''
                    for image_path in test:
                        X_test = prepareImages(image_path, 1, 'test')
                        predicted_test.append(gmodel.predict(np.array(X_test), verbose=1))
                    predicted_test = np.array(predicted_test)
                    predicted_test = np.reshape(predicted_test, (predicted_test.shape[0], predicted_test.shape[2]))
                    # ---------------여기까지
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
                print("1 OR 2 만 입력해주세요 ",err)

            except ValueError as err:
                print("only number :",err)


            except KeyboardInterrupt:
                print("retry")


if __name__ == "__main__":
    main()