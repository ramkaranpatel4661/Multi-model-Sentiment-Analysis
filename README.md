# Multi-modal Sentiment Analysis

## 🎭 Overview
This project implements a **Multi-modal Sentiment Analysis** system that fuses **text and image** data to predict the sentiment of social media posts. It utilizes a deep learning approach combining **DistilBERT** (for text) and **ResNet50** (for images) to achieve accurate sentiment classification.

## ✨ Features
*   **Multi-modal Fusion**: Combines transformer-based text embeddings with CNN-based image features.
*   **Interactive Web App**: A user-friendly Streamlit interface for real-time inference.
*   **Custom Dataset Support**: Designed for the Memotion Dataset 7k.
*   **Robust Architecture**: Built with PyTorch and Hugging Face Transformers.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/ramkaranpatel4661/Multi-model-Sentiment-Analysis.git
    cd Multi-model-Sentiment-Analysis
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

### Running the Web App
To start the interactive interface:
```bash
streamlit run app.py
```
Then open your browser at `http://localhost:8501`.

### Training the Model
To train the model from scratch on your dataset:
```bash
python src/train.py --epochs 3
```
This will save the best performing model as `best_model.pth`.

## 📂 Project Structure
```
Multi-modal Sentiment Analysis/
├── src/
│   ├── data_loader.py    # Dataset loading & preprocessing
│   ├── model.py          # Multi-modal Neural Network Architecture
│   ├── train.py          # Training loop & evaluation
│   └── test_run.py       # Verification script
├── app.py                # Streamlit Web Application
├── requirements.txt      # Project Dependencies
└── README.md             # Documentation
```

## 📊 Model Details
-   **Text Encoder**: DistilBERT (`distilbert-base-uncased`)
-   **Image Encoder**: ResNet50 (Pretrained on ImageNet)
-   **Fusion Strategy**: Concatenation of features -> Fully Connected Layers

## 📝 License
[MIT](https://choosealicense.com/licenses/mit/)
