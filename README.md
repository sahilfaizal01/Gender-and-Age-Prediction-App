# Gender-and-Age-Prediction-App
An app to predict the gender and age of people using CNN based deep learning model 

## Dataset:-
https://www.kaggle.com/datasets/jangedoo/utkface-new

## Concept:-
We will use CNN layers to extract features which are then taken as input by two different fully-connected(FC) layers where one FC does binary classification of a person's gender while the other predicts the age using regression methodology. Since classification and regression takes place here, two different evaluation metrics is used ie; for binary classification we rely on accuracy while for regression we monitor variation in loss values. Binary crossentropy loss is used for classification and mean squared error is used for regression.

## Image Dimension:
75 * 75 * 3 (RGB Image)

## Model Building:
* A Fine-tuning based approached is followed for extracting image features using Xception network
* Two different fully-connected layers are designed
* Gender Model - 128 Neurons in first layer with relu activation, 1 Neuron in output layer with sigmoid activation
* Age Model - 128 Neurons in first layer with relu activation, 1 Neuron in output layer with relu activation
* Optimizer - Adagrad

## Model Architecture:-
![image](https://user-images.githubusercontent.com/106440078/212733164-fa426d67-f57e-4680-b9fe-70654ac00f04.png)

## Model Summary:
![image](https://user-images.githubusercontent.com/106440078/212733367-8c793536-a280-4ec9-b53b-bcc2071c11d9.png)

## Model Weights:
https://drive.google.com/file/d/1VMPSHr6lxh2l5U_q808SA5eFLeNrFd5s/view?usp=share_link

## Accuracy and Loss Plot - Classification (Gender Classification):
### Accuracy vs Epoch Plot
![image](https://user-images.githubusercontent.com/106440078/212733610-1793cd0f-c6ce-4255-85c4-80a0db747ea9.png)
### Loss vs Epoch Plot
![image](https://user-images.githubusercontent.com/106440078/212733718-ed39fe7d-ad93-4b39-89a0-d78325935fdd.png)

## Loss Curve of Regression (Age Prediction):
![image](https://user-images.githubusercontent.com/106440078/212733770-26f6b08c-7362-4000-aa50-941d9e301ab2.png)

## Prediction Result:
![image](https://user-images.githubusercontent.com/106440078/212734014-9bff8679-5b12-4c27-9d0b-4c1b728e0a74.png)

## Conclusion:
* The accuracy and loss values are saturating near to 30 epochs
* Training Accuracy: 80% and Validation Accuracy: 79% (Gender Classification)
* Training Loss: ~4 and Validation Loss: ~72 (Age Prediction)
* The results can be further improved by using more data and by training for more epochs 
* Due to computational limitation I haven't experimented further:)

# Interface Design:
![image](https://user-images.githubusercontent.com/106440078/212739785-1d60fcf4-2d3e-4070-8ae1-07a348eeebe6.png)
