from PIL import Image

import numpy as np
import torch
import json
import os
import configs
import pandas as pd
import torchvision.transforms as T
from einops import rearrange, repeat


emb_dims = configs.embedding_dims

np.random.seed(20224343)
class ImageDataset:
    def __init__(self, batch_size):

        self.image_dir =r'D:\laion400m\images\\'
        self.embeddings_dir=r'D:\laion400m\embeddings\\'
        self.csvfile=r'D:\laion400m\laion_data_0.csv'
        self.df = pd.read_csv(self.csvfile)
        self.dataset_size=len(self.df)
        self.batch_size=batch_size
        sorted_index=np.array([i for i in range(0,self.dataset_size)])
        np.random.shuffle(sorted_index)
        self.unsorted_index=sorted_index
        assert (self.dataset_size% self.batch_size)==0,'dataset len is not devisiable by batch size'
        self.batches=np.reshape(self.unsorted_index,(-1,self.batch_size))
        #self.image_dims=(256,256)

        self.resolution=64
        self.M =self.resolution #self.image_dims[0]
        self.N =self.resolution #self.image_dims[1]

        self.totorch = T.Compose([T.ToTensor()])


    def get_item(self, idx):
        #load images from disc and load pre computed sentence embeddings

        imagepath=self.image_dir+str(idx)+'.png'
        embeddings_path=self.embeddings_dir+str(idx)+'.npy'

        image=Image.open(imagepath)
        image=self.preprocess_image(image)

        encoded_text=np.load(embeddings_path)[:, -1][0] #should be revisited
        encoded_text=torch.from_numpy(encoded_text)

        return image,encoded_text #,attention masks

    def preprocess_image(self, pil_image):

        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.Resampling.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y: crop_y + self.resolution, crop_x: crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1
        arr = self.totorch(arr)

        return arr


    def prepare_batch_on_mem(self,batch):
        batch_size=self.batch_size

        b_image_tensor = torch.zeros(batch_size,3,self.M,self.N)
        b_encoded_text=torch.zeros(batch_size,1,emb_dims)
        #b_attention_masks=torch.zeros(batch_size,8) # will see if it is needed for training

        for _,i in enumerate(batch):
            img,encoded_text=self.get_item(i)

            b_image_tensor[_,:,:,:]=img
            b_encoded_text[_,:,:]=encoded_text
            #b_attention_masks[_,:]=attention_masks
        #print(b_image_tensor.shape,encoded_text.shape)


        return b_image_tensor, b_encoded_text

    def get_batch(self,batch_number):

        batch=self.batches[batch_number]
        return self.prepare_batch_on_mem(batch)

    def pre_process_text_embeds(self,encoded_text):

        return encoded_text







