import cv2 as cv
import numpy as np
import os
import re
def natural_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def load_images_from_folder(folder):
    images = {}
    for filename in os.listdir(folder):
        category = []
        path = folder + "/" + filename
        for cat in sorted(os.listdir(path), key=natural_key):
            img = cv.imread(path + "/" + cat)
            if img is not None:
                category.append(img)
        images[filename] = category
    return images

images = load_images_from_folder(r'datasets')

def artefatos_sift(images):
    sift_vectors = {}
    descritor_lista = []
    sift = cv.SIFT_create()
    for k, value in images.items():
        features = []
        for img in value:
            kp, des = sift.detectAndCompute(img, None)
            descritor_lista.extend(des)
            features.append(des)
        sift_vectors[k] = features
    return [descritor_lista, sift_vectors]

sifts = artefatos_sift(images)
lista_descritores = sifts[0]
lista_features = sifts[1]

np.savez("lista_sift.npz", lista_descritores=lista_descritores)

data = np.load("lista_sift.npz")
print(data)
vocabulario = data["lista_descritores"]
