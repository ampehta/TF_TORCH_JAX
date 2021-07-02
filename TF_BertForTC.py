import transformers
import tensorflow as tf
import pandas as pd
from transformers import BertTokenizer,TFBertModel

bert = {'BERT:1':'kykim/bert-kor-base'}

class BertForClassification(tf.keras.Model):
    def __init__(self,huggingface_model_id):
        super(BertForClassification,self).__init__()
        self.model = TFBertModel.from_pretrained(huggingface_model_id)

    def call(self,x):
        x = self.model(x)['pooler_output']
        return x

def get_tokenizer_model(huggingface_model_id):
    tokenizer = BertTokenizer.from_pretrained(huggingface_model_id)
    model = BertForClassification(huggingface_model_id)
    return tokenizer,model  
  
if __name__ == '__main__':
    train = pd.read_csv('/content/drive/MyDrive/DACON:TopicClassificatiion/KLUE/train_data.csv')

    train_data = train['title'].values.tolist()
    train_target = train['topic_idx'].values.tolist()
    
    tokenizer, model = get_tokenizer_model(bert['BERT:1'])
    train_X = tokenizer.batch_encode_plus(train_data,padding='longest',return_tensors='tf')['input_ids']
