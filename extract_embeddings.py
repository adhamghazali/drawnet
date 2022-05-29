import pandas as pd
import T5
import numpy as np
from tqdm import tqdm
import os

csvfile = r'D:\laion400m\laion_data.csv'




def load_model():
    model_name = 't5-small'
    encoder = T5.t5encoder(model_name)
    return encoder

def extract_embeddings():
    return -1#extracted_embeddings

def save_to_file(embeddings,filename,embeddigns_folder):
    #np.save()
    return -1


if __name__ == "__main__":
    load_model()
    csvfile = r'D:\laion400m\laion_data_0.csv'
    embeddings_save_filename = r'D:\laion400m\embeddings\\'
    if not os.path.exists(embeddings_save_filename):
        os.makedirs(embeddings_save_filename)

    df = pd.read_csv(csvfile)

    for i in tqdm(range(len(df))):
        sentence = df['text'][i]
        print(sentence)





