import os
import cv2
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from tqdm import tqdm
import warnings
import shutil

warnings.filterwarnings("ignore")

# --- הגדרות (שנה את זה!) ---
# הנתיב לתיקייה הראשית שבה נמצאות 4 התיקיות (FakeVideo-FakeAudio וכו')
SOURCE_ROOT_DIR = r"C:\Users\reish\Desktop\FakeAvCeleb" 

# לאן לשמור את התוצאות (לתוך הפרויקט שלך)
OUTPUT_ROOT = "./dataset/train"

# שמור פריים אחד כל X פריימים (10 זה מאוזן)
FRAME_INTERVAL = 10 

def extract_frames(video_path, output_folder):
    """מחלץ פריימים מוידאו"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved_count = 0
    filename_prefix = os.path.splitext(os.path.basename(video_path))[0]
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # שמירה במרווחים
        if count % FRAME_INTERVAL == 0:
            save_path = os.path.join(output_folder, f"{filename_prefix}_frame{count}.jpg")
            try:
                cv2.imwrite(save_path, frame)
                saved_count += 1
            except Exception as e:
                pass
        count += 1
    cap.release()

def extract_spectrogram(video_path, output_folder):
    """יוצר ספקטרוגרמה מהאודיו"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filename_prefix = os.path.splitext(os.path.basename(video_path))[0]
    save_path = os.path.join(output_folder, f"{filename_prefix}_spec.png")
    
    if os.path.exists(save_path): return

    try:
        video_clip = VideoFileClip(video_path)
        if video_clip.audio is None: return
            
        audio_path = "temp_audio.wav"
        video_clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
        video_clip.close()
        
        y, sr = librosa.load(audio_path)
        plt.figure(figsize=(4, 4))
        plt.axis('off'); plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
        
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=sr)
        
        plt.savefig(save_path, bbox_inches=None, pad_inches=0)
        plt.close()
        
        if os.path.exists(audio_path): os.remove(audio_path)
    except:
        pass

def main():
    print(f"Starting processing from: {SOURCE_ROOT_DIR}")
    
    # מיפוי: איזה תיקיית מקור הולכת לאיזה לייבל (Real/Fake)
    # מבנה: 'שם_תיקייה_במקור': {'video': 'תווית_וידאו', 'audio': 'תווית_אודיו'}
    folder_map = {
        'RealVideo-RealAudio': {'video': 'Real', 'audio': 'Real'},
        'FakeVideo-FakeAudio': {'video': 'Fake', 'audio': 'Fake'},
        'FakeVideo-RealAudio': {'video': 'Fake', 'audio': 'Real'},
        'RealVideo-FakeAudio': {'video': 'Real', 'audio': 'Fake'}
    }

    # מעבר על התיקיות הראשיות
    for folder_name, labels in folder_map.items():
        source_path = os.path.join(SOURCE_ROOT_DIR, folder_name)
        
        if not os.path.exists(source_path):
            print(f"Skipping {folder_name} (Not found)")
            continue
            
        print(f"Processing {folder_name}...")
        
        # איסוף כל הסרטונים בתיקייה (גם בתתי-תיקיות)
        video_files = []
        for root, dirs, files in os.walk(source_path):
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mov')):
                    video_files.append(os.path.join(root, file))
        
        for video_path in tqdm(video_files):
            # הגדרת נתיבי יעד לפי הלוגיקה (Real/Fake)
            # FRAMES -> Real / Fake
            frames_target = os.path.join(OUTPUT_ROOT, "FRAMES", labels['video'])
            # SPECTOGRAMS -> Real / Fake
            specs_target = os.path.join(OUTPUT_ROOT, "SPECTOGRAMS", labels['audio'])
            
            # ביצוע החילוץ
            extract_frames(video_path, frames_target)
            extract_spectrogram(video_path, specs_target)

    print("\nProcessing Complete!")
    print(f"Data saved to: {os.path.abspath(OUTPUT_ROOT)}")

if __name__ == "__main__":
    main()