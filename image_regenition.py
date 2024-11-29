import os
from PIL import Image
import numpy as np
import csv
import pandas as pd 

# Load the image and convert to grayscale
image_path = "Img/img001-001.png"  # Replace with your image path
image = Image.open(image_path).convert("L")  # Convert image to grayscale
# data predection output 
data = pd.read_csv("english.csv")
Y = data["label"].values
# Convert the image to a numpy array
pixel_array = np.array(image).flatten()
print("Pixel Values as Array:")
print(pixel_array)
print("expect value is:", Y[100])
possible_values =  [0,1,2,3,4,5,6,7,8,9]
for i in range(0,26):
    possible_values.append(chr(i + ord("A")))
for i in range(0,26):
    possible_values.append(chr(i + ord("a")))
def disision_vector(V , Y_i):
    binary_vector = np.zeros(len(V), dtype=int)
    for i in range(len(V)):
       if V[i] ==Y_i:
            binary_vector[i] = 1
    return binary_vector
Normilised =  disision_vector(possible_values, Y[0])
print(Normilised)
