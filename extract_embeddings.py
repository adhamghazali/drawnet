import pandas as pd
import T5
import numpy as np
from tqdm import tqdm
import os

def load_model():
    model_name = 't5-small'
    encoder = T5.t5encoder(model_name)
    return encoder


if __name__ == "__main__":
    encoder=load_model()
    csvfile = r'D:\laion400m\laion_data_0.csv'
    embeddings_save_filename = r'D:\laion400m\embeddings\\'
    if not os.path.exists(embeddings_save_filename):
        os.makedirs(embeddings_save_filename)

    df = pd.read_csv(csvfile)

    for i in tqdm(range(len(df))):
        sentence = df['text'][i]
        fnx=embeddings_save_filename+str(i)
        encoded_text,masks=encoder.get_encoded_text(sentence,save_to_file=True)
        print("pre: ",encoded_text.shape)
        encoded_text, masks = map(lambda t: t.to('cpu'), (encoded_text, masks))
        print(encoded_text.shape)


        np.save(fnx, encoded_text.cpu())

