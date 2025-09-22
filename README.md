# Playing Cards Image Classifier

## 🃏 Project Overview
This project classifies images of playing cards into their respective categories using a Convolutional Neural Network (CNN).  
The model is trained on a dataset containing images of 52 standard playing cards and 1 joker card, totaling 53 classes.

## 📦 Dataset
The dataset used for training and evaluation is sourced from Kaggle:

**Dataset:** [Cards Image Dataset-Classification](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification)

**Details:**
- Total Images: 7,643 training images, 265 validation images, 265 test images
- Image Dimensions: 224x224 pixels
- Format: RGB
- Classes: 53 (52 standard playing cards + 1 joker)

## 🧠 Model Architecture
The classifier uses a Convolutional Neural Network (CNN) with:
- Convolutional layers for feature extraction
- MaxPooling layers for downsampling
- Dense layers for classification
- Softmax activation for multi-class output

## 🛠️ Training Process
### 1. Data Preprocessing
- Resize images to 224x224 pixels
- Normalize pixel values to [0, 1]

### 2. Model Training
- Loss Function: Categorical Crossentropy
- Optimizer: Adam
- Metrics: Accuracy
- Epochs: 10
- Batch Size: 32

### 3. Evaluation
- Evaluated on the test set
- Achieved accuracy: ~98%

## 🚀 Usage
### Gradio Demo
You can interact with the trained model via a Gradio interface:

[Launch Gradio Demo](https://huggingface.co/spaces/subhiksha-kodi/END-TO-END-CNN-PROJECT-CARDS-IMAGE-CLASSIFIER)

### Local Usage
1. Clone the repository:
```bash
git clone https://github.com/yourusername/playing-cards-image-classifier.git
cd playing-cards-image-classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Gradio interface:
```bash
python app.py
```

4. Open the provided local URL in your browser to interact with the model.

## 📂 Project Structure
```
.
├── app.py                     # Gradio interface code
├── cards_classifier.h5        # Trained CNN model
├── class_labels.json          # Class labels mapping
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
└── dataset/                   # Optional dataset folder
```

## 📄 License
This project is licensed under the MIT License.