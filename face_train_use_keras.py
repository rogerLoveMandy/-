#CNN网络模型类
import  random
from  keras.preprocessing.image import  ImageDataGenerator

from keras.layers import  Conv2D,MaxPooling2D,Activation,Dropout,Flatten,Dense,Convolution2D
from sklearn.model_selection  import  train_test_split

from  keras.models import  Sequential

from keras.optimizers import SGD

from  keras.utils import   np_utils

from  keras.models import  load_model

from load_face_dataset import load_dataset, resize_image, IMAGE_SIZE

from  keras import backend as k



class Model:
     def __init__(self):
         self.model = None

     #建立模型
     def build_model(self, dataset, nb_classes = 2):
         #构建一个空的网络模型，它是一个线性堆叠模型，各神经网络层会被顺序添加，专业名称为序贯模型或线性堆叠模型
         self.model = Sequential()

         self.modelmodel.add(Conv2D(
             filters=32,
             activation='relu',
             padding='same',
             input_shape=(IMAGE_SIZE, IMAGE_SIZE, img_channels),
             strides=(1, 1),
             name='Con2D1',
             kernel_size=(3, 3)
         ))

         self.model.add(MaxPooling2D(pool_size=(2, 2)))
         model.add(Dropout(0.5))

         self.model.add(Conv2D(
             filters=64,
             strides=(1, 1),
             padding='same',
             kernel_size=(3, 3),
             name='Con2D2',
             activation='relu'

         ))

         self.model.add(MaxPooling2D(pool_size=(2, 2)))

         self.model.add(Dropout(0.5))

         self.model.add(Flatten())

         self.model.add(Dense(128, activation='relu'))
         self.model.add(Dense(64, activation='relu'))
         self.model.add(Dense(32, activation='relu'))
         self.model.add(Dense(16, activation='relu'))

         self.model.add(Dense(8, activation='relu'))

         self.model.add(Dense(2, activation='softmax'))

         self.model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
         # 最后定义损失函数

         self.model.fit(self.train_images, self.train_labels, validation_data=(self.valid_inamges, self.valid_labels),
                   epochs=40, batch_size=25)
         # 最后放入批量样本，进行训练

         scores = self.model.evaluate(self.test_images, self.test_labels, verbose=0)

         self.model.save('model_weight.h5')

     # 识别人脸
     def face_predict(self, image):
         # 依然是根据后端系统确定维度顺序
         if k.image_dim_ordering() == 'th' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
             image = resize_image(image)
             image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))
         elif k.image_dim_ordering() == 'tf' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
             image = resize_image(image)
             image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))

         # 浮点并归一化
         image = image.astype('float32')
         image /= 255.0
         # 给出输入属于各个类别的概率，这里是二值类别，输出属于0和1的概率各为多少

         # 给出类别预测：0或者1
         result = self.model.predict_classes(image)
         print("result----"+ str(result))
         # 返回类别预测结果
         return result[0]

     # 训练模型
     # 训练模型
     def train(self, dataset, batch_size=25, nb_epoch=10, data_augmentation=True):
             sgd = SGD(lr=0.01, decay=1e-6,
                       momentum=0.9, nesterov=True)
             self.model.compile(loss='categorical_crossentropy',
                                optimizer=sgd,
                                metrics=['accuracy'])

             # 不使用数据提升，所谓的提升就是从我们提供的训练数据中利用旋转、翻转、加噪声等方法创造新的
             # 训练数据，有意识的提升训练数据规模，增加模型训练量
             if not data_augmentation:
                 self.model.fit(dataset.train_images,
                                dataset.train_labels,
                                batch_size=batch_size,
                                nb_epoch=nb_epoch,
                                validation_data=(dataset.valid_images, dataset.valid_labels),
                                shuffle=True)
             else:
                 datagen = ImageDataGenerator(
                     featurewise_center=False,
                     samplewise_center=False,
                     featurewise_std_normalization=False,
                     samplewise_std_normalization=False,
                     zca_whitening=False,
                     rotation_range=20,
                     width_shift_range=0.2,
                     height_shift_range=0.2,
                     horizontal_flip=True,
                     vertical_flip=False)

                 # 计算整个训练样本集的数量以用于特征值归一化、ZCA白化等处理
             datagen.fit(dataset.train_images)
             # 利用生成器开始训练模型
             self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels, batch_size=batch_size),
                                      samples_per_epoch=dataset.train_images.shape[0], nb_epoch=nb_epoch,
                                      validation_data=(dataset.valid_images, dataset.valid_labels))

     MODEL_PATH = 'me.face.model.h5'

     def save_model(self, file_path=MODEL_PATH):
         self.model.save(file_path)

     def load_model(self, file_path=MODEL_PATH):
         self.model = load_model(file_path)

     def evaluate(self, dataset):
         score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose=1)
         print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))


class Dataset:
    def __init__(self,path_name):
        self.train_images = None
        self.train_labels = None


        self.valid_inamges = None
        self.valid_labels = None


        self.test_images = None
        self.test_labels = None



        self.path_name  = path_name

        self.input_shape = None

        # 加载数据集并按照交叉验证的原则划分数据集并进行相关预处理工作
    def load(self,img_rows = IMAGE_SIZE,img_cols = IMAGE_SIZE,img_channels = 3 ,nb_classes = 2):
        #加载数据到内存
        images,labels = load_dataset(self.path_name)

        train_images,valid_images,train_labels,valid_labels  = \
            train_test_split(images,labels,test_size=0.3,random_state=random.randint(0,100))


        _,test_images,_,test_labels = train_test_split(images,labels,test_size=0.5,random_state=random.randint(0,100))





        # 当前的维度顺序如果为'th'，则输入图片数据时的顺序为：channels,rows,cols，否则:rows,cols,channels
        # 这部分代码就是根据keras库要求的维度顺序重组训练数据集

        if  k.image_dim_ordering() == 'th':
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)



        # 我们的模型使用categorical_crossentropy作为损失函数，因此需要根据类别数量nb_classes将
        # 类别标签进行one-hot编码使其向量化，在这里我们的类别只有两种，经过转化后标签数据变为二维

        train_labels =  np_utils.to_categorical(train_labels,nb_classes)
        valid_labels =  np_utils.to_categorical(valid_labels,nb_classes)

        test_labels =  np_utils.to_categorical(test_labels,nb_classes)

        # 像素数据浮点化以便归一化
        train_images = train_images.astype('float32')
        valid_images = valid_images.astype('float32')
        test_images = test_images.astype('float32')


        #将其归一化,图像的各像素值归一化到0~1区间

        train_images/=255
        valid_images/=255
        test_images/=255
        #数据集先浮点后归一化的目的是提升网络收敛速度，减少训练时间，同时适应值域在（0,1）之间的激活函数，
        # 增大区分度。其实归一化有一个特别重要的原因是确保特征值权重一致。举个例子，我们使用mse这样的均方误差函数时，
        # 大的特征数值比如(5000-1000)2与小的特征值(3-1)2相加再求平均得到的误差值，显然大值对误差值的影响最大，但大部分情况下，
        # 特征值的权重应该是一样的，只是因为单位不同才导致数值相差甚大。因此，我们提前对特征数据做归一化处理，以解决此类问题。


        self.train_images = train_images
        self.train_labels = train_labels

        self.valid_images = valid_images
        self.valid_labels = valid_labels

        self.test_images = test_images
        self.test_labels = test_labels



if __name__ == '__main__':
    dataset = Dataset('D:/tmp/data/')
    dataset.load()
    model = Model()
    model.build_model(dataset)
    model.train(dataset)
    model.save_model(file_path='model_weight.h5')

