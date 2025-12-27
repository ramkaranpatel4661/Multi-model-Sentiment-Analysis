import torch
from model import MultiModalSentimentModel

def test_forward_pass():
    print("Initializing model...")
    model = MultiModalSentimentModel(num_classes=5)
    model.eval()
    
    # Dummy data
    batch_size = 2
    seq_len = 128
    
    # DistilBERT inputs
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    mask = torch.ones((batch_size, seq_len))
    
    # ResNet inputs (batch, 3, 224, 224)
    pixel_values = torch.randn(batch_size, 3, 224, 224)
    
    print("Running forward pass...")
    try:
        output = model(input_ids, mask, pixel_values)
        print("Output shape:", output.shape) # Should be (2, 5)
        print("Success! Model works.")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_forward_pass()
