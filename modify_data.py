import imutils
import numpy as np
import cv2 as cv
from jitter import jitterImage
from imutils import paths
import csv

cascade_faces = "resources\\haarcascade_frontalface_default.xml";
face_detection = cv.CascadeClassifier(cascade_faces);

image_paths_angry = list(paths.list_images('fer2013\\train\\Angry'))
image_paths_disgust = list(paths.list_images('fer2013\\train\\Disgust'))
image_paths_fear = list(paths.list_images('fer2013\\train\\Fear'))
image_paths_happy = list(paths.list_images('fer2013\\train\\Happy'))
image_paths_neutral = list(paths.list_images('fer2013\\train\\Neutral'))
image_paths_sad = list(paths.list_images('fer2013\\train\\Sad'))
image_paths_surprise = list(paths.list_images('fer2013\\train\\Surprise'))

f = open("new_fer2013.csv", "a")
writer = csv.writer(f)

for image_path in image_paths_angry:
    image = cv.imread(image_path)
    new_images = None

    faces = face_detection.detectMultiScale(image, scaleFactor = 1.2,
        minNeighbors = 5, minSize = (20, 20))
    
    if(len(faces) > 0):
        new_images = jitterImage(image, faces)

    if(new_images != None):
        for new_image in new_images:
            imagem_cinza = cv.cvtColor(new_image, cv.COLOR_BGR2GRAY)
            img2 = imutils.resize(imagem_cinza, width=48, height=48)

            lin_img = np.array(img2).reshape(-1)
            listToStr = ' '.join([str(elem) for elem in lin_img])

            data = [0, listToStr, "Training"]          
            writer.writerow(data)
         
    new_images = None
    image = None
    faces = None
