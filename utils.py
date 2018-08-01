import numpy as np
from PIL import Image
import requests
from io import BytesIO



def read_image_from_s3(s3_url):
    """
    Read images into memory given an s3 url.
    Returns a list of images.
    """
    imgs = []
    for url in s3_url:
        response = requests.get(url)
        img = np.array(Image.open(BytesIO(response.content)))
        imgs.append(img)
        
    return np.array(imgs)