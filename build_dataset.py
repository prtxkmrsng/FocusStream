import os
import cv2
import pandas as pd
from biometrics import extract_live_biometrics

def process_image_folder(folder_path, state_label):
    extracted_data = []
    success_count = 0
    skip_count = 0
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            frame = cv2.imread(image_path)
            
            if frame is None:
                continue
                
            # Use our new MediaPipe sensors!
            _, features = extract_live_biometrics(frame)
            
            if features is not None:
                row = features + [state_label]
                extracted_data.append(row)
                success_count += 1
            else:
                skip_count += 1
                
    print(f"   -> Successfully tracked: {success_count} faces (Skipped {skip_count} bad images)")
    return extracted_data

def main():
    print("[Data Pipeline] Starting MediaPipe precision extraction...")
    all_data = []
    base_dir = "raw_data"
    
    categories = {"focused": 0, "fatigued": 1}
    
    for category_name, state_label in categories.items():
        folder_path = os.path.join(base_dir, category_name)
        if os.path.exists(folder_path):
            print(f"\nProcessing folder: {category_name} (Label: {state_label})")
            folder_data = process_image_folder(folder_path, state_label)
            all_data.extend(folder_data)

    columns = ['ear', 'blink_duration', 'mar', 'head_deviation', 'state']
    df = pd.DataFrame(all_data, columns=columns)
    
    df.to_csv("drowsiness_data.csv", index=False)
    print(f"\n✅ Success! New precision dataset generated with {len(df)} rows.")

if __name__ == "__main__":
    main()