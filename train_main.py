import argparse
import os
import torch.multiprocessing

# --- תיקון Imports למבנה הפרויקט ---
# מנסה לטעון את הפונקציות מהמיקום הנכון בתוך Multimodal
try:
    from Multimodal.utils.Common_Function import set_seeds
    from Multimodal.Unimodal.Training import Train_MesoNet, Train_MesoInceptionNet, Train_Xception, Train_EfficientB0, Train_VGG16, Train_F3Net
except ImportError as e:
    print(f"Import Error: {e}")
    print("Trying alternative import (if folders were moved)...")
    # Fallback למקרה שהתיקיות הוזזו
    from utils.Common_Function import set_seeds
    from Unimodal.Training import Train_MesoNet, Train_MesoInceptionNet, Train_Xception, Train_EfficientB0, Train_VGG16, Train_F3Net

# --- הגדרת נתיבים דינמיים ---
current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = current_script_path
dataset_train_root = os.path.join(project_root, 'dataset', 'train')
models_save_root = os.path.join(project_root, 'models')

# יצירת תיקיית מודלים אם לא קיימת
os.makedirs(models_save_root, exist_ok=True)

# ברירות מחדל לנתיבים (ניתן לשינוי דרך ה-Terminal)
default_video_path = os.path.join(dataset_train_root, 'FRAMES')
default_audio_path = os.path.join(dataset_train_root, 'SPECTOGRAMS')

parser = argparse.ArgumentParser(description='PyTorch CONTINUAL LEARNING')
parser.add_argument('--model', '-m', type=str, default="MESO4", 
                    help='model name = [MESO4, MESOINCEPTION4, XCEPTION, EFFICIENTB0, F3NET, LIPS, XRAY]')
parser.add_argument('--path_video', '-v', type=str, default=default_video_path, 
                    help='path of video frames')
parser.add_argument('--path_audio', '-a', type=str, default=default_audio_path, 
                    help='path of audio spectrograms')
parser.add_argument('--path_save', '-sm', type=str, default=models_save_root, 
                    help='path to save model while training')
parser.add_argument('--lr', '-l', type=float, default=1e-5, help='initial learning rate')
parser.add_argument('--epochs', '-me', type=int, default=50, help='epochs')
parser.add_argument('--batch_size', '-nb', type=int, default=32, help='batch size (lowered to avoid OOM)')
parser.add_argument('--num_gpu', '-ng', type=str, default='0', help='gpu number')
parser.add_argument('--val_ratio', '-vr', type=float, default=0.3, help='validation ratio on trainset')
parser.add_argument('--n_early', '-ne', type=int, default=10, help='patience number of early stopping')

args = parser.parse_args()
set_seeds()
torch.multiprocessing.set_sharing_strategy('file_system')

print(f'Running on GPU: {args.num_gpu}')
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.num_gpu)
MODEL = args.model

print(f"--- Starting Training ---")
print(f"Model: {MODEL}")
print(f"Video Path: {args.path_video}")
print(f"Save Path: {args.path_save}")
print(f"-------------------------")

if MODEL == 'MESO4':
    Train_MesoNet.TrainMesoNet(args)
elif MODEL == 'MESOINCEPTION4':
    Train_MesoInceptionNet.TrainMesoInceptionNet(args)
elif MODEL in ['XCEPTION', 'XRAY', 'LIPS']:
    if MODEL == 'LIPS':
        args.path_video = args.path_video.replace('FRAMES', 'FRAMES_LIPS')
    if MODEL == 'XRAY':
        args.path_video = args.path_video.replace('FRAMES', 'FRAMES_XRAY')     
    Train_Xception.TrainXception(args)
elif MODEL == 'EFFICIENTB0':
    Train_EfficientB0.TrainEfficientB0(args)
elif MODEL == 'VGG':
    Train_VGG16.TrainVGG16(args)
elif MODEL == 'F3NET':
    Train_F3Net.TrainF3Net(args)
else:
    print(f"Error: Model {MODEL} not recognized.")