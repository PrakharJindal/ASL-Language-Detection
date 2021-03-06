# ASL Sign Language Detection

### Introduction

In this project, we are using a deep learning based model for detecting American Sign Language. Here we have got Validation accuracy of 89.74 %.

### I/O Screenshots

- Detecting Alphabet L
 
![Alpha L Detecting](https://github.com/PrakharJindal/ASL-Language-Detection/blob/main/media/Prediction_alpha_L(Small).png)

- Detecting Alphabet O
 
![Alpha O Detecting](https://github.com/PrakharJindal/ASL-Language-Detection/blob/main/media/Prediction_alpha_O(Small).png)

- Flow Chart
 
![Flow Chart](https://github.com/PrakharJindal/ASL-Language-Detection/blob/main/media/flowchart(Small).png)

### Using the application:

The website uses OpenCV to access the webcam. The image frames recieved are processed for the model. We use cvZone module for Hand detection which in backend uses OpenCV and Mediapipe . A box is generated around hand and the predicted aplhabet is printed along with bounding box.

### Dataset

The dataset includes 29 folders ( this include 26 alphabets and 3 action for delete , nothing, space )

### Model Generation

MobileNet ( with Non-Trainable Layers) with extra fully connected layers is being used and a H5 file is generated.

### Running the application locally

- Install python ,django, mediapipe, keras , CVZone , Tensorflow and opencv 
- Clone the repo
- Go to the project root dir
- open the root folder of the project in cmd
- run command : python manage.py runserver
