import tensorflow as tf
import tensorflow_addons as tfa 

class gmlp_block(tf.keras.Model):
    def __init__(self,d_model,d_ffn):
        super(gmlp_block,self).__init__()

        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.channel_proj_i = tf.keras.layers.Dense(d_ffn)
        self.channel_proj_ii = tf.keras.layers.Dense(d_model)
        self.spatial_gating_unit = spatial_gating_unit(d_ffn)

    def call(self,x):
        residual = x
        x = self.layer_norm(x)
        x = tfa.activations.gelu(self.channel_proj_i(x))

        x = self.spatial_gating_unit(x)
        x = self.channel_proj_ii(x)

        return x + residual

class spatial_gating_unit(tf.keras.Model):
    def __init__(self,d_ffn):
        super(spatial_gating_unit,self).__init__()

        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.spatial_proj = tf.keras.layers.Conv1d(d_ffn/2,1,kernel_initializer='zeros',bias_initializer='ones')
        
    def call(self,x):
        u,v = tf.split(x, 2, axis=-1)
        v = self.layer_norm(v)
        v = self.spatial_proj(v)
        return u * v 
    
class gMLP(tf.keras.Model):
    def __init__(self,d_model=256,d_ffn=512,seq_len=256,num_layers=6):
        super(gMLP,self).__init__()
        self.model = tf.keras.Sequential([gmlp_block(d_model,d_ffn,seq_len) for _ in range(num_layers)])

    def call(self,x):
        x = self.model(x)
        return x

class gMLP_LanguageModel(gMLP):
    def __init__(self,num_tokens=10000, d_model=256, d_ffn=512, seq_len=256, num_layers=6):
        super().__init__(d_model,d_ffn,seq_len)
        self.embed = tf.keras.layers.Embedding(num_tokens,d_model)
        
    def call(self,x):
        embedding = self.embed(x)
        output = self.model(embedding)
        return output
