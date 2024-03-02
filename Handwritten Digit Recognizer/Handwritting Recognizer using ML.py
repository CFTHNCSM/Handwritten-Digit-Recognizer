#Machine Learning Mini Project
#Name - D.M.A.S. Dasanayaka


#First download the below libraries to your IDE by inserting these commands to the terminal.
#pip install tensorflow
#pip install matplotlib
#pip install numpy
#pip install cv

#################     Importing the relavent directories     ################     
import os
import cv2 #computer vision to load images
import numpy as np #to work with the numpy arrays
import matplotlib.pyplot as plt #visualization of the digits
import tensorflow as tf #to have the training and testing of the model





################     Designing the model      ################     
#delclaring the relavant variables
mnist = tf.keras.datasets.mnist #we can load the dataset from MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data() #x-handwirtter digits, y-classification

x_train = tf.keras.utils.normalize(x_train, axis=1)    #scaling it down so every value is in  binary
x_test = tf.keras.utils.normalize(x_test, axis=1)

#Creating the model
model = tf.keras.models.Sequential()  #basic sequential neural network
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))   # it is to flatten the input into a single line, here it is 28*28 digit line. 28*28 means the number of pixels. Pictures with higher pixels values will take more time to process.
model.add(tf.keras.layers.Dense(128, activation='relu'))  #rectify linear unit - relu
model.add(tf.keras.layers.Dense(128, activation='relu'))  
model.add(tf.keras.layers.Dense(10, activation='softmax'))  

#complining the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  #compiling the model with different attributes







#################      Saving and loading the model     ################     
model_file = r#Insert the location where the model needs to be saved.       Ex - 'C:\Users\msi gl63\Desktop\New folder'  #The path to save the model
# if os.path.exists(model_file):
#     model = tf.keras.models.load_model(model_file)
#     print("Model loaded successfully!")
# else:
model.fit(x_train, y_train, epochs=3) # Here we mention the number of iteration to train the model. High number of epochs results high accuracy but if it is very high model will be overfit.
model.save(model_file)
print("Model trained and saved successfully!")



#This is to check whether the model was saved at the correct location
print(model_file)






################      Performance Evaluation     ################     
loss, accuracy = model.evaluate(x_test, y_test)
print(loss)
print(accuracy)







################     testing of the data   ################
# Set the folder path where the images are located, here we tell the program to check for the image folder

image_folder = r#Insert the folder location where the testing digits are located.        Ex - "C:\Users\Desktop\New folder\digits"

image_number = 1
image_path = os.path.join(image_folder, f"digit{image_number}.png") #After finding the folder, the image location can be found
# print(f"Attempting to load image from: {image_path}")
# img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

while os.path.isfile(image_path):
    try:
        img = cv2.imread(image_path)[:,:,0]
        img = np.invert(np.array([img])) #The image is as an numpy array, this is convert the array to image.
        prediction = model.predict(img)
        print(f"this digit is probably a {np.argmax(prediction)}") #This is to find the highest activated neuron in the network.
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except Exception as e:
        print(f'Error processing {image_path}: {e}')
    finally:
        image_number += 1
        image_path = os.path.join(image_folder, f"digit{image_number}.png")


print("All the images have been predicted.")   #To verify that al the numbers are read by the model.



