🎭 Multi-Modal Sentiment Analysis — Text + Image Fusion










A deep-learning based multi-modal sentiment analysis system that predicts sentiment by combining text and image features. The project uses:

📝 DistilBERT (Transformers) for text embeddings

🖼️ ResNet-50 (CNN) for image feature extraction

🔗 Feature Fusion + Classifier for sentiment prediction

🌐 Streamlit Web App for real-time inference

Trained on the Memotion Dataset 7K.

🚀 Features

🔹 Multi-modal Fusion — Transformer (Text) + CNN (Image)

🔹 Pretrained Models — Transfer Learning for better accuracy

🔹 Real-time Prediction UI — Upload meme images & text

🔹 Modular Architecture — Clean and extensible project structure

🔹 Supports Custom Input & Dataset Extensions

📂 Project Structure
Multi-modal Sentiment Analysis/
├── memotion_dataset_7k/          # Dataset (images + labels)
├── src/
│   ├── data_loader.py            # Dataset loader & preprocessing
│   ├── model.py                  # Multi-modal model architecture
│   ├── train.py                  # Training & validation pipeline
│   └── utils.py                  # Helper utilities
├── app.py                        # Streamlit Web App
├── requirements.txt              # Dependencies
└── README.md                     # Documentation

🧠 Model Architecture
Component	Technique
Text Encoder	DistilBERT (distilbert-base-uncased)
Image Encoder	ResNet-50 (Pretrained ImageNet)
Fusion	Concatenation of Text + Image embeddings
Classifier	Fully-Connected Layers
Output	Sentiment Class (Positive / Neutral / Negative)
🛠 Installation
git clone https://github.com/ramkaranpatel4661/Multi-model-Sentiment-Analysis.git
cd Multi-model-Sentiment-Analysis
pip install -r requirements.txt

🧪 Train the Model
python src/train.py --epochs 3


The best model will be saved as:

best_model.pth

🌐 Run the Web App
streamlit run app.py


Open in browser:

http://localhost:8501

📊 Future Enhancements (Planned)

🔸 Support for emotion classification

🔸 Attention-based fusion layer

🔸 Explainable AI visualization for prediction insights

🔸 Model performance dashboard

🖼 Demo Preview (Add Screenshots Here)

📌 Add images like:

Training results

Streamlit app output

Sample predictions

/assets/screenshots/app_demo.png
/assets/screenshots/results.png

🤝 Contributing

Pull requests are welcome!
Feel free to open an Issue or submit an Improvement Suggestion.

📜 License

This project is licensed under the MIT License.

⭐ Support

If you like this project:

👉 Star the repo on GitHub
👉 Share or contribute 🙂