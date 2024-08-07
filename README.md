# Gesture-Recognition-using-Convolutional-Neural-Networks-CNNs-
Overview
This project demonstrates the implementation of a Convolutional Neural Network (CNN) for recognizing hand gestures using the LeapGestRecog dataset. The CNN model is designed to classify images into one of ten gesture categories, achieving high accuracy through effective preprocessing, training, and evaluation.

Dataset
The LeapGestRecog dataset contains images of hand gestures, categorized into ten classes:

Palm
L
Fist
Fist Moved
Thumb
Index
Ok
Palm Moved
C
Down
Preprocessing
Image Loading: Images are loaded in grayscale and resized to 50x50 pixels.
Normalization: Image pixel values are normalized to the range [0, 1].
One-Hot Encoding: Labels are one-hot encoded for model training.
Data Splitting: The dataset is split into training and testing sets.
Model Architecture
The CNN model consists of the following layers:

Convolutional Layers with ReLU activation and MaxPooling.
Dropout layers to prevent overfitting.
Flatten layer to convert 2D features to 1D.
Dense layers with ReLU and Softmax activation for classification.
Training
The model is compiled using the categorical_crossentropy loss function and rmsprop optimizer. It is trained for 7 epochs with a batch size of 32, using 70% of the data for training and 30% for validation.

Evaluation
The model's performance is evaluated using accuracy and loss metrics on the test set. Confusion matrices and learning curves are plotted to visualize the results.

Results
Accuracy: The model achieved a test accuracy of over 90%.
Confusion Matrix: A confusion matrix is plotted to show the classification performance across different gesture categories.
Learning Curves: Plots of training and validation loss/accuracy over epochs.
Usage
To run the code, ensure you have the necessary libraries installed (keras, tensorflow, opencv, numpy, matplotlib, seaborn). Load the dataset and execute the script to train and evaluate the model.

Future Work
Experiment with deeper CNN architectures.
Implement data augmentation to enhance model generalization.
Deploy the model for real-time gesture recognition applications.
Conclusion
This project demonstrates the effectiveness of CNNs in recognizing hand gestures with high accuracy. By leveraging the LeapGestRecog dataset and appropriate preprocessing techniques, the model can reliably classify gestures into predefined categories.

Acknowledgments
The creators of the LeapGestRecog dataset.
Open-source libraries and the developer community for providing tools and resources.
