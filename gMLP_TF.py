import tensorflow as tf
import tensorflow_addons as tfa 

class gmlp_block(tf.keras.Model):
    def __init__(self,d_model,d_ffn):
        super(gmlp_block,self).__init__()

        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.channel_proj_i = tf.keras.layers.Dense(d_model)
        self.channel_proj_ii = tf.keras.layers.Dense(d_ffn)
        self.spatial_gating_unit = spatial_gating_unit()

    def call(self,x):
        residual = x
        x = self.layer_norm(x)
        x = tfa.activations.gelu(self.channel_proj_i(x))

        x = self.spatial_gating_unit(x)
        x = self.channel_proj_ii(x)

        return x + residual

class spatial_gating_unit(tf.keras.Model):
    def __init__(self):
        super(spatial_gating_unit,self):

        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.spatial_proj = 
        
    def call(self,x):
        u,v = tf.split(x, 2, axis=-1)
        v = self.layer_norm(v)
        v = self.spatial_proj(v)
        return u * v 
