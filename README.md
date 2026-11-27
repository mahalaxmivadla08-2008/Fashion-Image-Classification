# Fashion Image Classification using CNN

## Project Overview
This project uses Deep Learning and Convolutional Neural Networks (CNN) to classify fashion images into 10 categories using the Fashion-MNIST dataset.

## Objectives
- Build a CNN model for image classification.
- Train the model using Fashion-MNIST.
- Evaluate model performance.
- Create an interactive website for clothing predictions.

## Dataset
Fashion-MNIST dataset containing 70,000 grayscale images across 10 clothing categories.

## Technologies Used
- Python
- TensorFlow and Keras
- NumPy
- Matplotlib
- Scikit-learn
- Gradio

## Model Architecture
- Convolutional layers
- Max Pooling layers
- Flatten layer
- Dense layer
- Dropout
- Softmax output layer

## Project Files
- `app.py` – Web application
- `requirements.txt` – Required libraries
- `Fashion_Image_Classification.ipynb` – Model training notebook
- `training_results.json` – Actual training results
- `model/fashion_mnist_cnn.keras` – Trained CNN model

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Open the local Gradio URL displayed in the terminal.

## Results
Refer to `training_results.json` and the notebook for actual model evaluation results.

## Limitations
The model is trained on Fashion-MNIST grayscale images. Predictions on ordinary real-world clothing photographs may be less accurate.

## Future Improvements
- Improve performance on real-world clothing images.
- Add more image categories.
- Deploy the application online.

## Conclusion
This project demonstrates how CNN models can be used for fashion image classification through an interactive web application.
