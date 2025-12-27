<h1 align="center">🎭 Multi-Modal Sentiment Analysis</h1>
<p align="center">
🔗 Text + Image Fusion | 🤖 Deep Learning | 📝 DistilBERT | 🖼️ ResNet-50 | 🌐 Streamlit
</p>

<p align="center">
A deep-learning based multi-modal sentiment analysis system that predicts sentiment by combining <b>text</b> and <b>image</b> features.  
The model fuses DistilBERT text embeddings with ResNet-50 image features and performs feature-level fusion for sentiment classification.  
Built with <b>PyTorch</b> and <b>Hugging Face Transformers</b>, trained on the <b>Memotion Dataset 7K</b>.
</p>

---

## 🚀 Features
✔️ Multi-modal fusion — Transformer (Text) + CNN (Image)  
✔️ Pretrained models — Transfer Learning for improved accuracy  
✔️ Real-time prediction using Streamlit web app  
✔️ Modular architecture and clean project structure  
✔️ Supports custom input and dataset extensions  

---

## 🧠 Model Architecture

| Component | Technique |
|--------|--------|
| **Text Encoder** | DistilBERT (`distilbert-base-uncased`) |
| **Image Encoder** | ResNet-50 (Pretrained on ImageNet) |
| **Fusion Strategy** | Concatenation of Text + Image embeddings |
| **Classifier** | Fully-Connected Layers |
| **Output** | Sentiment Class — Positive / Neutral / Negative |

---

## 📂 Project Structure
```
Multi-modal Sentiment Analysis/
├── memotion_dataset_7k/      # Dataset (images + labels)
├── src/
│   ├── data_loader.py        # Dataset loader & preprocessing
│   ├── model.py              # Multi-modal model architecture
│   ├── train.py              # Training & validation pipeline
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
Best model will be saved as:
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
🔸 Emotion classification support  
🔸 Attention-based fusion layer  
🔸 Explainable AI visualizations  
🔸 Performance analytics dashboard  

---

## 🖼 Demo Preview (add screenshots later)
```
/assets/screenshots/app_demo.png
/assets/screenshots/results.png
```

---

## 🤝 Contributing
Pull requests are welcome — feel free to open an Issue or Suggestion.

---

## 📜 License
This project is licensed under the **MIT License**.

---

## ⭐ Support
If you find this project useful:

👉 Star the repo  
👉 Share or contribute 🙂
