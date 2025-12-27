import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from transformers import AutoTokenizer
from src.model import MultiModalSentimentModel
import os

# Page Config
st.set_page_config(page_title="Multi-modal Sentiment Analysis", page_icon="🎭")

# Device Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Model
@st.cache_resource
def load_model():
    model = MultiModalSentimentModel(num_classes=5)
    model_path = 'best_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded trained model.")
    else:
        print("Warning: 'best_model.pth' not found. Using untrained model.")
    
    model.to(device)
    model.eval()
    return model

# Load Tokenizer
@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0) # Add batch dim

def preprocess_text(text, tokenizer, max_len=128):
    inputs = tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        return_token_type_ids=True,
        truncation=True
    )
    return {
        'ids': torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0),
        'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(0)
    }

# Mapping
label_map = {
    0: 'Very Negative',
    1: 'Negative',
    2: 'Neutral',
    3: 'Positive',
    4: 'Very Positive'
}

# UI
st.title("🎭 Multi-modal Sentiment Analysis")
st.markdown("Analyze sentiment from **Text** and **Image** posts combined!")

col1, col2 = st.columns(2)

with col1:
    st.header("Input")
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    user_text = st.text_area("Enter Text", "Write something about the image...")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_container_width=True)

with col2:
    st.header("Prediction")
    if st.button("Analyze Sentiment"):
        if uploaded_file is None or not user_text.strip():
            st.error("Please provide both an image and text.")
        else:
            with st.spinner('Analyzing...'):
                # Load resources
                model = load_model()
                tokenizer = load_tokenizer()
                
                # Preprocess
                img_tensor = preprocess_image(image).to(device)
                text_data = preprocess_text(user_text, tokenizer)
                input_ids = text_data['ids'].to(device)
                mask = text_data['mask'].to(device)
                
                # Inference
                with torch.no_grad():
                    outputs = model(input_ids, mask, img_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    pred_label = torch.argmax(probs, dim=1).item()
                
                # Result
                sentiment = label_map[pred_label]
                confidence = probs[0][pred_label].item()
                
                st.success(f"Sentiment: **{sentiment}**")
                st.info(f"Confidence: {confidence:.2%}")
                
                # Chart
                st.bar_chart({label_map[i]: probs[0][i].item() for i in range(5)})

st.markdown("---")
st.caption("Powered by DistilBERT & ResNet50 | Trained on Memotion Dataset")
