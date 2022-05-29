import numpy as np
import pandas as pd
import os
import requests
import shutil
import csv
from tqdm import tqdm


def download_url(url, filename):
    try:
        res = requests.get(url, stream=True)
    except:
        return False

    if res.status_code == 200:

        with open(filename, 'wb') as f:
            shutil.copyfileobj(res.raw, f)

        return True
        # print('Image sucessfully Downloaded: ',file_name)
    else:
        return False

thepart=0

name=r'D:\archive'
parts=[os.path.join(name,i) for i in os.listdir(name)]
df = pd.read_parquet(parts[thepart])

pathname = r'D:\laion400m\images_' + str(thepart)

if not os.path.exists(pathname):
    os.makedirs(pathname)

rows = []
csvfile = r'D:\laion400m\laion_data_' + str(thepart) + '.csv'

with open(csvfile, 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['imagepath','text'])

    for i in tqdm(range(0, 10)):  # to len(df)
        url = df["URL"][i]
        idx = str(i) + '.png'  # need to figure out a unique index
        filename = os.path.join(pathname, idx)

        if download_url(url, filename):
            text = df["TEXT"][i]
            row = [filename, text]
            writer.writerow(row)

            # rows.append(row)
        else:
            print('unable to download url')
            continue
f.close()

