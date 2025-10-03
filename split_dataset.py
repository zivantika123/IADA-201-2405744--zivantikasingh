import os
import shutil
from sklearn.model_selection import train_test_split

def create_train_val_splits(base_dir, train_ratio=0.7):
    os.makedirs('dataset/train', exist_ok=True)
    os.makedirs('dataset/val', exist_ok=True)
    
    for class_name in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_name)
        if os.path.isdir(class_path) and not class_name.startswith('dataset'):
            os.makedirs(f'dataset/train/{class_name}', exist_ok=True)
            os.makedirs(f'dataset/val/{class_name}', exist_ok=True)
            
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if len(images) == 0:
                print(f"Skipping {class_name}: no images found.")
                continue  # Skip empty folders
            
            train_images, val_images = train_test_split(
                images, train_size=train_ratio, random_state=42
            )
            
            for img in train_images:
                shutil.copy2(os.path.join(class_path, img), os.path.join('dataset/train', class_name, img))
            for img in val_images:
                shutil.copy2(os.path.join(class_path, img), os.path.join('dataset/val', class_name, img))
            
            print(f"Processed {class_name}: {len(train_images)} train, {len(val_images)} validation")

if __name__ == "__main__":
    create_train_val_splits('.')
    print("Dataset split complete!")