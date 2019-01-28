import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from imgaeclassifier.Resnet import Resnet
from imgaeclassifier.SENet import SE_Resnet
from imgaeclassifier.DenseNet import DenseNet

def prepareImages(data, m, dataset):
    X_train = np.zeros((m, 100, 100, 3))
    count = 0
    for fig in data['Image']:
        # load images into images of size 100x100x3
        img = image.load_img("E:/Data/Humpback Whale Identification/" + dataset + "/" + fig, target_size=(100, 100, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)
        X_train[count] = x
        if (count+1) % 1000 ==0:
            print(count+1,": ",fig)
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
'''
def batch_generator(X, y, batch_size=16):
    count = 0
    while True:
        # choose batch_size random images / labels from the data
        idx = np.random.randint(0, X.shape[0], batch_size)
        X_t = X.loc[idx,'Image']
        X_train = prepareImages(X_t, X_t.shape[0], 'train/Augment')
        X_train /= 255
        X_train = np.array(X_train)
        label = y[idx]
        yield X_train, label
        count += 1
'''
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
    y, label_encoder = preprocessing_Label(train_df['Id'])
    file_path = ".model_weight.hdf5"
    callbacks = get_callbacks(file_path, patience=5)
    #model = Alexnet(input_shape=(224,224,3), num_classes=y.shape[1])
    #model = VGG('E', input_shape=(224,224,3), num_classes=y.shape[1])
    #model = Resnet(input_shape=(224,224,3), num_classes=y.shape[1])
    #model = SE_Resnet(input_shape=(224,224,3), num_classes=y.shape[1])
    model = DenseNet(input_shape=(100, 100, 3), num_classes=5005)
    gmodel = model.model


    while True:
            try:

                Input = int(input("You can select number. Training[1] / Test&Save[2] / Exit[3]"))
                if Input == 1:

                    X = prepareImages(train_df, train_df.shape[0], "train")
                    # X /= 255     # Why?
                    history = gmodel.fit(X, y,
                           epochs=20,
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
                    plt.plot(history.history['categorical_accuracy'])
                    plt.title('Model categorical accuracy')
                    plt.ylabel('categorical accuracy')
                    plt.xlabel('Epoch')
                    plt.show()

                elif Input == 2:
                    test = os.listdir(img_test_path+"/")
                    col = ['Image']
                    test_df = pd.DataFrame(test, columns=col)
                    print(len(test))
                    test_df['Id'] = ''
                    X = prepareImages(test_df, test_df.shape[0], "test")
                    X /= 255
                    gmodel.load_weights(file_path)
                    predictions = gmodel.predict(np.array(X), verbose=1)
                    for i, pred in enumerate(predictions):
                        test_df.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))
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