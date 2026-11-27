
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained CNN model
model = tf.keras.models.load_model(
    "model/fashion_mnist_cnn.keras"
)

class_names = [
    "T-shirt / Top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
]

def predict_fashion(image):
    if image is None:
        return "Please upload an image.", {}

    # Convert image to grayscale and resize
    image = image.convert("L").resize((28, 28))

    # Normalize image
    image_array = np.array(image).astype("float32") / 255.0

    # Reshape for CNN
    image_array = image_array.reshape(1, 28, 28, 1)

    # Predict
    predictions = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[predicted_index]) * 100

    result = (
        f"Prediction: {predicted_class}\n"
        f"Confidence: {confidence:.2f}%"
    )

    scores = {
        class_names[i]: float(predictions[i])
        for i in range(10)
    }

    return result, scores

with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 👕 Fashion Image Classification
        ### Deep Learning powered clothing recognition

        Upload a clothing image to predict its category.
        """
    )

    with gr.Row():
        image_input = gr.Image(
            type="pil",
            label="Upload Clothing Image"
        )

        with gr.Column():
            output_text = gr.Textbox(
                label="AI Prediction",
                lines=3
            )

            output_scores = gr.Label(
                num_top_classes=5,
                label="Prediction Confidence"
            )

    predict_button = gr.Button(
        "Predict Clothing Category",
        variant="primary"
    )

    predict_button.click(
        fn=predict_fashion,
        inputs=image_input,
        outputs=[output_text, output_scores]
    )

    gr.Markdown(
        "Built using TensorFlow, CNN and Fashion-MNIST dataset."
    )

demo.launch()
