# TF_TORCH_JAX
Of the many publicly available ML/DL frameworks I personally prefer using TensorFlow mostly because of the convenience Keras API provides. Recently however, I have seen the necessity to study additional frameworks such as Torch or Jax/Flax for the sake of my flexibility in learning and implementations.   

This repository will be recording my week long private programming session. I am planning to present 5 diffenent projects starting from simple classification problems to complex NLP modeling in all three frameworks(Tensorflow,Torch,Jax/Flax). Hopefully, I might be able to introduce my own implementation of a byte-to-byte Korean language model written in Jax/Flax by the end of the week.  
  
  ## Project 1: MNIST Classification. 
  Since it is my first time with Pytorch and Jax/Flax I started out with a comparably simple task. In case of Pytorch I made my own simple classifier and successfully ran in it on Cuda. Jax/Flax however, I had a hard time making the script to run. Flax.optim api was deprecated so I replaced it with optax and eventually had the model to train iteself. The training took about 30 minutes which is surpisingly long for a simple classification task, and I am almost sure that it was trained on a CPU instead of a cuda environment. I was just happy to see the script work and gave up fixing the problems. 
