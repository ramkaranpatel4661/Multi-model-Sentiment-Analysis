import torch
import torch.nn as nn
import torchvision.models as models
from transformers import DistilBertModel

class MultiModalSentimentModel(nn.Module):
    def __init__(self, num_classes=5):
        super(MultiModalSentimentModel, self).__init__()
        
        # Text Encoder (DistilBERT)
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.text_drop = nn.Dropout(0.3)
        
        # Image Encoder (ResNet50)
        weights = models.ResNet50_Weights.DEFAULT
        resnet = models.resnet50(weights=weights)
        # Remove the last FC layer
        modules = list(resnet.children())[:-1]
        self.image_encoder = nn.Sequential(*modules)
        self.image_drop = nn.Dropout(0.3)
        
        # Fusion Layer
        # DistilBERT hidden size: 768
        # ResNet50 output size: 2048
        self.fusion_dim = 768 + 2048
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, input_ids, attention_mask, pixel_values):
        # Text features
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # We use the CLS token representation (first token) - usually hidden_state[:, 0]
        # or last_hidden_state associated with the first token. 
        # DistilBERT output: last_hidden_state of shape (batch_size, sequence_length, hidden_size)
        text_features = text_output.last_hidden_state[:, 0, :]
        text_features = self.text_drop(text_features)
        
        # Image features
        image_features = self.image_encoder(pixel_values)
        # ResNet output shape: (batch_size, 2048, 1, 1), need to flatten
        image_features = image_features.view(image_features.size(0), -1)
        image_features = self.image_drop(image_features)
        
        # Fusion
        fused_features = torch.cat((text_features, image_features), dim=1)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits
