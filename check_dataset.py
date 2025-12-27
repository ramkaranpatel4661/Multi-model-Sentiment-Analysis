
import pandas as pd
import os

try:
    # Load labels
    csv_path = 'memotion_dataset_7k/labels.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        exit(1)
        
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from labels.csv")
    print("Columns:", df.columns.tolist())

    # Check image folder
    img_dir = 'memotion_dataset_7k/images'
    if not os.path.exists(img_dir):
        print(f"Error: {img_dir} not found.")
        exit(1)
        
    # Check image existence
    found_count = 0
    missing_images = []
    
    # Cleaning image names if necessary (sometimes there are weird spaces)
    df['image_name'] = df['image_name'].astype(str).str.strip()
    
    for img_name in df['image_name']:
        img_path = os.path.join(img_dir, img_name)
        if os.path.exists(img_path):
            found_count += 1
        else:
            missing_images.append(img_name)
            
    print(f"\nImages found: {found_count}/{len(df)}")
    if missing_images:
        print(f"Missing images (first 5): {missing_images[:5]}")
        
    # Check labels
    if 'overall_sentiment' in df.columns:
        print("\nLabel Distribution (overall_sentiment):")
        print(df['overall_sentiment'].value_counts())
    else:
        print("\n'overall_sentiment' column not found.")

except Exception as e:
    print(f"An error occurred: {e}")
