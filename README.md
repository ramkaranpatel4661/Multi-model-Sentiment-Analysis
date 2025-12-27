<h1 align="center">🎭 Multi-Modal Sentiment Analysis — Text + Image Fusion</h1>

<p align="center">
<img src="https://img.shields.io/badge/Python-3.10%2B-blue">
<img src="https://img.shields.io/badge/Deep%20Learning-Neural%20Networks-red">
<img src="https://img.shields.io/badge/HuggingFace-Transformers-yellow">
<img src="https://img.shields.io/badge/PyTorch-CNN-orange">
<img src="https://img.shields.io/badge/Streamlit-Web%20App-brightgreen">
</p>



<p align="center">
A deep-learning based multi-modal sentiment analysis system that analyzes social media posts by combining <b>text</b> and <b>image</b> data.  
The model extracts <b>text features using a Transformer (BERT / DistilBERT)</b> and <b>image features using a pre-trained CNN (ResNet-50)</b>, fuses the embeddings, and predicts sentiment labels such as <b>Positive, Neutral, or Negative</b>.
</p>

---

## 📌 Project Overview

This project follows the core objective of **Multi-Modal Sentiment Analysis**:

- 📝 Extract **text embeddings** using Transformer models  
  (BERT / DistilBERT — Hugging Face Transformers)
- 🖼 Extract **image embeddings** using a pre-trained CNN  
  (ResNet-50 — TorchVision / PyTorch)
- 🔗 **Fuse multi-modal embeddings** and train a classifier
- 📊 Evaluate performance on a **labeled multi-modal dataset**
- 🌐 Provide a **Streamlit web interface** for real-time prediction

The project is trained on the **Memotion Dataset 7K**, which contains meme images paired with text and sentiment labels.

---

## 🧰 Technologies & Tools

- **Python**
- **PyTorch / TorchVision**
- **Hugging Face Transformers (BERT / DistilBERT)**
- **Streamlit** (Web Interface)
- *(Optional)* TensorFlow / Flask
- **Public Multi-Modal Sentiment Datasets** (Memotion 7K)

---

## 🧱 Key Requirements Implemented

- Extract text features using **Transformer models**
- Extract image features using **pre-trained CNN**
- Perform **multi-modal feature fusion**
- Train a **sentiment classification model**
- Evaluate on a **labeled dataset**
- Provide a **simple web UI for inference**

---

## 📦 Deliverables

- 🧹 Data preprocessing scripts  
- 🤖 Multi-modal training & evaluation code  
- 🌐 Streamlit web application  
- 📊 Model performance results  
- 🖼 Demo screenshots (UI & predictions)

---

## 🧠 Model Architecture

| Component | Technique |
|--------|--------|
| **Text Encoder** | DistilBERT (`distilbert-base-uncased`) |
| **Image Encoder** | ResNet-50 (Pretrained on ImageNet) |
| **Fusion Strategy** | Concatenation of Text + Image embeddings |
| **Classifier** | Fully-Connected Layers |
| **Output** | Sentiment — Positive / Neutral / Negative |

---

## 📂 Project Structure

```
Multi-modal Sentiment Analysis/
├── memotion_dataset_7k/      # Dataset (images + labels)
├── src/
│   ├── data_loader.py        # Dataset loader & preprocessing
│   ├── model.py              # Multi-modal model architecture
│   ├── train.py              # Training & evaluation pipeline
│   └── utils.py              # Helper utilities
├── app.py                    # Streamlit Web App
├── requirements.txt          # Dependencies
└── README.md                 # Documentation
```

---

## 🛠 Installation

```bash
git clone https://github.com/ramkaranpatel4661/Multi-model-Sentiment-Analysis.git
cd Multi-model-Sentiment-Analysis
pip install -r requirements.txt
```

---

## 🧪 Train the Model

```bash
python src/train.py --epochs 3
```

Best model gets saved as:

```
best_model.pth
```

---

## 🌐 Run the Web App

```bash
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 📊 Future Enhancements

- 🔸 Emotion & sarcasm classification  
- 🔸 Attention-based fusion network  
- 🔸 Explainable-AI visualization  
- 🔸 Cross-dataset generalization experiments  

---

## 🖼 Demo Preview (Screenshots Placeholder)

```
/assets/screenshots/app_demo.png
/assets/screenshots/results.png
```

(Add screenshots after training & testing)

---

## 🤝 Contributing

Pull requests are welcome.  
Feel free to open an **Issue** or submit an **Improvement Suggestion**.

---

## 📜 License

This project is licensed under the **MIT License**.

---

## ⭐ Support

If you like this project:

👉 Star the repository  
👉 Share it  
👉 Contribute 🙂
