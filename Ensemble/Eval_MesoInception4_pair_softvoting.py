import os
from PIL import Image
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.multiprocessing
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import sklearn.metrics as metrics

# הנחת יסוד: הקבצים הללו קיימים בתיקייה שלך
from Common_Function_ import *
# שינוי: טעינת MesoInception4
from models.MesoNet4_forEnsemble import MesoInception4 as MesoNet

# --- הגדרות חומרה ---
torch.multiprocessing.set_sharing_strategy('file_system')
GPU = '0' # תוקן לכרטיס יחיד
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
pretrained_size = 224
pretrained_means = [0.4489, 0.3352, 0.3106]
pretrained_stds = [0.2380, 0.1965, 0.1962]

# --- Dataset Class ---
class CustumDataset(Dataset):
    def __init__(self, data, target, data_2=None, target_2=None, transform=None):
        self.data = data
        self.target = target
        self.data_video = data_2
        self.target_video = target_2
        self.transform = transform

        if self.data_video:
            self.len_data2 = len(self.data_video)
        
        print(f"Data Video Len: {self.len_data2}")
        print(f"Target Len: {len(self.target)}")

        assert (self.len_data2 == len(self.target) == len(self.target_video) == len(self.data) == len(self.data_video))

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        path = self.data[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        if self.data_video:
            path_video = self.data_video[idx]
            img_video = Image.open(path_video).convert('RGB')
            if self.transform:
                img_video = self.transform(img_video)
            return img, self.target[idx], img_video, self.target_video[idx]
            
        return img, self.target[idx]

def getnum_of_files(path):
    _dict = {}
    for (root, dirs, files) in os.walk(path):
        if not dirs:
            _dict[root.split(os.path.sep)[-1]] = len(files)
    return _dict

# --- Ensemble Class (Soft Voting Logic) ---
class Ensemble(nn.Module):
    def __init__(self, models=[], device='cuda', training=True):
        super().__init__()
        self.model1 = None
        self.model2 = None
        assert len(models) >= 2
        if models:
            self.model1 = models[0]  # FOR VIDEO
            self.model2 = models[1]  # FOR FRAME IMAGE

    def forward(self, frame, video):
        # הערה: הסדר כאן תלוי במימוש המקורי. 
        # בקוד המקורי feat1 מקבל video ו-feat2 מקבל frame.
        feat1 = self.model1(video)
        feat2 = self.model2(frame)
        out = feat1 + feat2
        out = out / 2 # ממוצע (Soft Voting)
        return out

# --- תיקון נתיבים דינמי ---
current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_path)

test_dir = [
    os.path.join(project_root, "dataset", "test", "SPECTOGRAMS", "real_A_fake_others"),
    os.path.join(project_root, "dataset", "test", "FRAMES", "real_A_fake_others")
]

checkpoint_dir = os.path.join(project_root, "models")
MODELS_NAME = 'MesoInception4'

test_transforms = transforms.Compose([
    transforms.Resize((pretrained_size, pretrained_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=pretrained_means, std=pretrained_stds)
])

# טעינת הדאטה
print(f"Loading data from: {test_dir}")
list_test = [datasets.ImageFolder(root=test_dir[0], transform=None),
             datasets.ImageFolder(root=test_dir[1], transform=None)]

# הכנת רשימות קבצים
list_glob_testpath = [list_test[1].samples[i][0] for i in range(len(list_test[1].samples))]
list_targets_testpath = [list_test[1].targets[i] for i in range(len(list_test[1].targets))]

list_num_test = getnum_of_files(test_dir[1])
list_glob_testpath_video = []
list_targets_testpath_video = []

for i in range(len(list_test[0].samples)):
    folder_name = list_test[0].samples[i][0].split(os.path.sep)[-2]
    num_repeat = int(list_num_test.get(folder_name, 0))
    list_glob_testpath_video += [list_test[0].samples[i][0]] * num_repeat
    list_targets_testpath_video += [list_test[0].targets[i]] * num_repeat

assert(list_targets_testpath_video == list_targets_testpath)
test_data = CustumDataset(list_glob_testpath, list_targets_testpath, list_glob_testpath_video, list_targets_testpath_video, test_transforms)
print(f'Number of testing examples: {len(test_data)}')


# --- טעינת מודלים ---
models = [MesoNet(), MesoNet()]

path_model_1 = os.path.join(checkpoint_dir, f'{MODELS_NAME}_realA_fakeB.pt')
path_model_2 = os.path.join(checkpoint_dir, f'{MODELS_NAME}_realA_fakeC.pt')

print(f"Loading models from {checkpoint_dir}...")
list_checkpoint = [
    torch.load(path_model_1, map_location=device)['state_dict'],
    torch.load(path_model_2, map_location=device)['state_dict']
]

models[0].load_state_dict(list_checkpoint[0])
models[1].load_state_dict(list_checkpoint[1])

# יצירת ה-Ensemble
ecls = Ensemble(models, device)
if torch.cuda.device_count() > 1:
    ecls = nn.DataParallel(ecls)
ecls = ecls.to(device)
ecls.eval()

# --- הרצת הערכה (Evaluation) ---
test_iterator = DataLoader(test_data, shuffle=True, batch_size=BATCH_SIZE)

y_true = []
y_pred_probs = [] # הסתברויות למחלקה 1 (Fake)
y_pred_labels = [] # חיזוי סופי (0 או 1)

print("Starting Soft Voting Evaluation...")
for i, batch_data in enumerate(test_iterator):
    with torch.no_grad():
        # batch_data: [img_audio, target_audio, img_video, target_video]
        in_1 = batch_data[0].to(device) # Audio
        in_2 = batch_data[2].to(device) # Video
        targets = batch_data[1].to(device)
        
        # הרצת ה-Ensemble (מחזיר וקטור הסתברויות לכל תמונה)
        outputs = ecls(in_1, in_2) 
        
        # שמירת התוצאות
        y_true.extend(targets.cpu().numpy())
        
        # Softmax כדי לקבל הסתברויות יפות (אם המודל לא מוציא אותן כבר)
        # ב-MesoNet הפלט הוא בד"כ אחרי Softmax או LogSoftmax, אבל נניח שזה Logits
        # אם זה כבר הסתברות, ה-argmax יעבוד אותו דבר.
        
        probs = torch.nn.functional.softmax(outputs, dim=1)
        y_pred_probs.extend(probs[:, 1].cpu().numpy()) # הסתברות להיות Fake
        y_pred_labels.extend(outputs.argmax(dim=1).cpu().numpy())

# המרה למערכי Numpy
y_true = np.array(y_true)
y_pred_labels = np.array(y_pred_labels)
y_pred_probs = np.array(y_pred_probs)

# --- הדפסת תוצאות ---
print("-" * 30)
print(f'Final Accuracy: {accuracy_score(y_true, y_pred_labels) * 100:.2f}%')
print("-" * 30)

print(classification_report(y_true, y_pred_labels, digits=5, zero_division='warn'))

# --- ROC Plot ---
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred_probs)
roc_auc = metrics.auc(fpr, tpr)

plt.figure()
lw = 2
print('ROC AUC: {:.3f}'.format(roc_auc))
plt.plot(fpr, tpr, color='darkred', lw=lw, label='ROC curve ({:.3f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Ensemble Soft Voting ({MODELS_NAME})')
plt.legend(loc="lower right")

save_path = os.path.join(project_root, 'roc_curve_soft_inception.png')
plt.savefig(save_path)
print(f"ROC curve saved to {save_path}")