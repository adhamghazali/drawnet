from PIL import Image

import numpy as np
import torch
import json
import os
import configs
import pandas as pd

emb_dims = configs.embedding_dims

np.random.seed(2022)
class ImageDataset:
    def __init__(self, batch_size):

        self.image_dir =r'D:\laion400m\images'
        self.embeddings_dir=r'D:\laion400m\embeddings'
        self.csvfile=r'D:\laion400m\laion_data_0.csv'
        self.df = pd.read_csv(self.csvfile)
        self.dataset_size=len(self.df)
        self.batch_size=batch_size
        sorted_index=np.array([i for i in range(0,self.dataset_size)])
        np.random.shuffle(sorted_index)
        self.unsorted_index=sorted_index
        assert (self.dataset_size% self.batch_size)==0,'dataset len is not devisiable by batch size'
        self.batches=np.reshape(self.unsorted_index,(-1,self.dataset_size))[0]
        print(self.batches)




    def get_item(self, idx):
        #TODO: load images from disc and load pre computed sentence embeddings



        return image,encoded_text, attention_masks

    def generate_random_batch(self):
        #needs more work,
        #TODO: incorporate higher prioroties for unvisted indexes such that we make sure to cover the dataset
        return np.random.randint(0,self.dataset_size,self.batch_size)

    def prepare_batch_on_mem(self,batch):
        batch_size=self.batch_size

        b_image_tensor = torch.zeros(batch_size,3,M,N)
        b_encoded_text=torch.zeros(batch_size,8,emb_dims)
        b_attention_masks=torch.zeros(batch_size,8)


        for i,_ in enumerate(batch):
            img,encoded_text,attention_masks=get_item(i)
            b_image_tensor[i,:,:,:]=img
            b_encoded_text[i,:,:]=encoded_text
            b_attention_masks[i,:]=attention_masks

        return b_image_tensor, b_encoded_text,b_attention_masks
    def get_batch(self,batch_number):

        batch=self.batches[batch_number]
        return self.prepare_batch_on_mem(batch)



X=ImageDataset(2)




