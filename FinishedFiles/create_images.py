import os
import pandas as pd
import requests

df = pd.read_csv("lookalikes_16_classes_dataset.csv")
os.makedirs("plant_images", exist_ok=True)

'''
test_url = "https://inaturalist-open-data.s3.amazonaws.com/photos/272309/medium.JPG"
response = requests.get(test_url)
with open("foo.jpg", "wb") as f:
    f.write(response.content)
'''

for index, row in df.iterrows():
    if index <= 28328:
        continue
    else:
        url = row["image_url"]
        response = requests.get(url)
        with open(f'plant_images/{index}.jpg', 'wb') as f:
            f.write(response.content)

df['image_path'] = [f'plant_images/{index}.jpg' for index in range(len(df))]
df.to_csv("lookalikes_16_classes_dataset.csv",index=False)