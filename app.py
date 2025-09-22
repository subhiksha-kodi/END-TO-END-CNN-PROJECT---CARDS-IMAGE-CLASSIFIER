import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json

# Load model and labels
model = load_model("cards_classifier.h5")
with open("class_labels.json", "r") as f:
    class_labels = json.load(f)
class_labels = {int(k): v for k, v in class_labels.items()}

def predict_card(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    label = class_labels[predicted_class]
    return f"🔮 Predicted Card: {label}\n✨ Confidence: {confidence*100:.2f}%"

# Custom CSS (Glassmorphism + Neon)
custom_css = """
body, .gradio-container {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    font-family: 'Poppins', sans-serif;
    color: #fff !important;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
}

/* Header */
.header {
    text-align: center;
    margin-bottom: 25px;
}
.header h1 {
    font-size: 28px;
    color: #fdfdfd;
    text-shadow: 0px 0px 10px rgba(255,255,255,0.6);
}
.header p {
    color: #ddd;
    font-size: 15px;
}

/* Glass card */
.glass-panel {
    backdrop-filter: blur(18px) saturate(180%);
    -webkit-backdrop-filter: blur(18px) saturate(180%);
    background: rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.25);
    padding: 30px;
    max-width: 480px;
    margin: auto;
    box-shadow: 0 8px 25px rgba(0,0,0,0.4);
    animation: fadeIn 1s ease;
}

/* Upload box */
.gr-image {
    border-radius: 15px !important;
    border: 2px dashed rgba(255,255,255,0.4) !important;
    padding: 10px;
    background: rgba(255,255,255,0.05);
}

/* Button glow */
.gr-button {
    background: linear-gradient(90deg, #ff6ec4, #7873f5) !important;
    border: none !important;
    color: white !important;
    font-weight: bold !important;
    border-radius: 12px !important;
    padding: 12px 20px !important;
    margin-top: 18px !important;
    box-shadow: 0 0 12px rgba(255,110,196,0.6);
    transition: 0.25s ease-in-out;
}
.gr-button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 22px rgba(120,115,245,0.9);
}

/* Output */
.output-box {
    margin-top: 20px;
    background: rgba(255,255,255,0.1) !important;
    border-radius: 15px;
    padding: 15px;
    font-size: 17px;
    font-weight: 500;
    color: #fff !important;
    border: 1px solid rgba(255,255,255,0.25);
    animation: slideUp 0.8s ease;
    text-align: center;
}

/* Animations */
@keyframes fadeIn {
    from {opacity:0; transform: scale(0.95);}
    to {opacity:1; transform: scale(1);}
}
@keyframes slideUp {
    from {opacity:0; transform: translateY(15px);}
    to {opacity:1; transform: translateY(0);}
}
"""

# Gradio UI
with gr.Blocks(css=custom_css) as demo:
    with gr.Column(elem_classes="header"):
        gr.Markdown("<h1>🌌 Futuristic Card Classifier</h1>")
        gr.Markdown("<p>Upload your playing card and see the glowing prediction ✨</p>")

    with gr.Column(elem_classes="glass-panel"):
        input_img = gr.Image(type="pil", label="Upload a Card")
        predict_btn = gr.Button("🔮 Reveal Prediction")
        output_text = gr.Textbox(label="Result", elem_classes="output-box")

        predict_btn.click(fn=predict_card, inputs=input_img, outputs=output_text)

demo.launch()
