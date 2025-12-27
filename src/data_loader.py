import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MemotionDataset(Dataset):
    def __init__(self, csv_file, img_dir, tokenizer, max_len=128, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform
        
        # specific label mapping
        self.label_map = {
            'very_negative': 0,
            'negative': 1,
            'neutral': 2,
            'positive': 3,
            'very_positive': 4
        }
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Image
        img_name = str(self.data.iloc[idx]['image_name']).strip()
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except (OSError, FileNotFoundError):
            # Fallback for missing or corrupt images (though we verified them, good practice)
            # Create a black image
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)
        else:
            # Default transform if none provided
            default_tf = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])
            image = default_tf(image)

        # Text
        text = str(self.data.iloc[idx]['text_corrected'])
        if text == 'nan':
            text = ""
            
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        
        # Label
        label_str = self.data.iloc[idx]['overall_sentiment']
        # Handle cases where label might be null or unexpected? 
        # The verification showed valid labels, but let's be safe.
        label = self.label_map.get(label_str, 2) # Default to neutral if not found
        
        return {
            'image': image,
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }
