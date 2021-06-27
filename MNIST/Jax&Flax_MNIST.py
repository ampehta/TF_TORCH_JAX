import jax
import jax.numpy as jnp

from flax import linen as nn 
from flax import optim
import optax

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

def normalize(data,target):
    return tf.cast(data,tf.float32)/255.,target
  
class mJAX(nn.Module):
    @nn.compact
    def __call__(self,x):
        x = x.reshape((x.shape[0],-1))
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dropout(0.2,deterministic=True)(x) #deterministic=True가 없는경우 오류 왜인지 잘 모르겠음
        x = nn.Dense(10)(x)
        x = nn.softmax(x)
        return x
      
def cross_entropy_loss(pred_logits,target):
    one_hot_target = jax.nn.one_hot(target,num_classes=10)
    return -jnp.mean(jnp.sum(pred_logits*one_hot_target,axis=-1))

def compute_metrics(logits, labels):
  loss = cross_entropy_loss(logits, labels)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy
  }
  return metrics

def train_step(optimizer,apply_fun,batch,opt_state,params,logits):
    def loss_obj(params): # params == logits
        logits = apply_fun({'params':params},batch['data']) # updated_state
        loss = cross_entropy_loss(logits,batch['target'])
        return loss,logits

    (_,logits) , grad = jax.value_and_grad(loss_obj,has_aux=True)(params)
    updates , opt_state = optimizer.update(grad,opt_state)
    params = optax.apply_updates(params,updates)

    metrics = compute_metrics(logits,batch['target'])
    return opt_state,params,logits,metrics

  
if __name__ == '__main__':
  (train,test),info = tfds.load('MNIST',split=['train','test'],as_supervised=True,with_info=True)

  train = train.map(normalize)
  train_ds = train.batch(32).prefetch(tf.data.AUTOTUNE)
  
  mJax = mJAX()
  init_shape = jnp.ones((32, 28, 28, 1), jnp.float32)
  logits,params = mJax.init(jax.random.PRNGKey(0),init_shape).pop('params')

  optimizer = optax.adam(0.001)
  opt_state = optimizer.init(params)
  
  epochs = 6

  for epoch in range(epochs):
    train_loss, train_accuracy = [] , []
    for data,target in train_ds:
        batch = {'data':jnp.float32(data.numpy()),
                 'target':jnp.float32(target.numpy())}

        opt_state,params,logits,metrics = train_step(optimizer,mJax.apply,batch,opt_state,params,logits)

        train_loss.append(metrics['loss'])
        train_accuracy.append(metrics['accuracy'])

    print(f'Epoch: {epoch+1} | Loss: {np.mean(train_loss)} | Accuracy: {np.mean(train_accuracy)*100}')
  
