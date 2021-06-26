import tensorflow as tf
import tensorflow_datasets as tfds


def normalize_dataset(image,label):
    return tf.cast(image,tf.float32)/255. ,  # 0~255 사이의 숫자로 이루어진 데이터셋 정규화
  
class MNIST_TF(tf.keras.Model):
    def __init__(self):
        super(MNIST_TF,self).__init__()
        self.model = tf.keras.Sequential([
                                          tf.keras.layers.Flatten(input_shape=(28,28)),
                                          tf.keras.layers.Dense(128,activation='relu'),
                                          tf.keras.layers.Dropout(0.2),
                                          tf.keras.layers.Dense(10,activation='softmax')
        ])
    def call(self,x):
        x = self.model(x)
        return x
      
      

if __name__ == '__main__':
  device = tf.test.gpu_device_name()
  print(device)
  
  (train,test) , info = tfds.load('mnist',split=['train','test'],shuffle_files=True, as_supervised=True, with_info=True)
  # as_supervised returns the value in a tuple, if False it returns values in a dictionary

  train_ds = train.map(normalize_dataset).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
  test_ds = test.map(normalize_dataset).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
  
  mTF = MNIST_TF()
  opt = tf.keras.optimizers.Adam(0.001)
  mTF.compile(optimizer=opt,loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
  #from_logits = True only when not using proper final activation functions (softmax, sigmoid) 
  
  mTF.fit(train_ds,epochs=6)
  mTF.evaluate(test_ds)
