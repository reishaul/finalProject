import os
from PIL import Image
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.multiprocessing
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import sklearn.metrics as metrics

# הנחת יסוד: הקבצים הללו קיימים בתיקייה שלך
from Common_Function_ import *
from models.MesoNet4_forEnsemble import Meso4 as MesoNet

# --- הגדרות חומרה וסביבה ---
torch.multiprocessing.set_sharing_strategy('file_system')
GPU = '0'  # תוקן מ-'0,1,2' לכרטיס יחיד
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 50
BATCH_SIZE = 64
VALID_RATIO = 0.3
N_IMAGES = 100
START_LR = 1e-5
END_LR = 10
NUM_ITER = 100
PATIENCE_EARLYSTOP = 10

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
        
        # הדפסות ביקורת
        print(f"Data Video Len: {self.len_data2}")
        print(f"Target Len: {len(self.target)}")

        assert (self.len_data2 == len(self.target) == len(self.target_video) == len(self.data) == len(self.data_video))

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Audio Spectrogram
        path = self.data[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        # Video Frame
        if self.data_video:
            path_video = self.data_video[idx]
            img_video = Image.open(path_video).convert('RGB')
            if self.transform:
                img_video = self.transform(img_video)
            return img, self.target[idx], img_video, self.target_video[idx]
            
        return img, self.target[idx]

train_transforms = transforms.Compose([
    transforms.Resize((pretrained_size, pretrained_size)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=pretrained_means, std=pretrained_stds)
])

test_transforms = transforms.Compose([
    transforms.Resize((pretrained_size, pretrained_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=pretrained_means, std=pretrained_stds)
])

def getnum_of_files(path):
    _dict = {}
    for (root, dirs, files) in os.walk(path):
        if not dirs:
            _dict[root.split(os.path.sep)[-1]] = len(files)
    return _dict

# --- תיקון נתיבים דינמי ---
current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_path)

test_dir = [
    os.path.join(project_root, "dataset", "test", "SPECTOGRAMS", "real_A_fake_others"),
    os.path.join(project_root, "dataset", "test", "FRAMES", "real_A_fake_others")
]

checkpoint_dir = os.path.join(project_root, "models")
MODELS_NAME = 'Meso4'

# טעינת שמות הקבצים ל-Dataset
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

# --- Ensemble Class ---
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
        feat1 = self.model1(video) # שימו לב: כאן הוחלפו השמות בטעות בקוד המקורי או שזה מכוון, שמרתי על הלוגיקה המקורית
        feat2 = self.model2(frame)
        out = feat1 + feat2
        out = out / 2
        return out

# --- טעינת מודלים ---
models = [MesoNet(), MesoNet()]

path_model_1 = os.path.join(checkpoint_dir, f'{MODELS_NAME}_realA_fakeB.pt')
path_model_2 = os.path.join(checkpoint_dir, f'{MODELS_NAME}_realA_fakeC.pt')

print("Loading Checkpoints...")
list_checkpoint = [
    torch.load(path_model_1, map_location=device)['state_dict'],
    torch.load(path_model_2, map_location=device)['state_dict']
]

models[0].load_state_dict(list_checkpoint[0])
models[1].load_state_dict(list_checkpoint[1])

# הגדרת המודל המאוחד
ecls = Ensemble(models, device)
if torch.cuda.device_count() > 1:
    ecls = nn.DataParallel(ecls)
ecls = ecls.to(device)

# --- הרצת ה-Test ---
label_encoder = LabelEncoder()
enc = OneHotEncoder(sparse=False)

y_true = np.zeros((0, 2), dtype=np.int8)
y_pred = np.zeros((0, 2), dtype=np.int8)
y_true_auc = []
y_pred_auc = []

ecls.eval()
test_iterator = data.DataLoader(test_data, shuffle=True, batch_size=BATCH_SIZE)

print("Starting Evaluation...")
for i, batch_data in enumerate(test_iterator):
    with torch.no_grad():
        in_1 = batch_data[0].to(device) # Audio/Spectrogram
        in_2 = batch_data[2].to(device) # Video/Frame
        
        # Forward pass
        _y_pred = ecls(in_1, in_2).cpu().detach()

        # שמירה ל-AUC
        _pred = copy.deepcopy(_y_pred).detach().cpu()
        _true = copy.deepcopy(batch_data[1]).detach().cpu().float().tolist()
        
        for _a in _pred[:, 1]: y_pred_auc.append(_a)
        for _a in _true: y_true_auc.append(_a)

        # עיבוד ה-Labels (לוגיקה מקורית)
        integer_encoded = label_encoder.fit_transform(batch_data[1].detach().cpu())
        integer_encoded_2 = label_encoder.fit_transform(batch_data[3].detach().cpu())
        
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        integer_encoded_2 = integer_encoded_2.reshape(len(integer_encoded_2), 1)

        onehot_encoded = enc.fit_transform(integer_encoded)
        onehot_encoded_2 = enc.fit_transform(integer_encoded_2)
        
        onehot_encoded = onehot_encoded.astype(np.int8)
        onehot_encoded_2 = onehot_encoded_2.astype(np.int8)

        # שילוב ה-Labels ובניית ה-True Values
        _y_true = torch.tensor(onehot_encoded + onehot_encoded_2)
        _y_true_argmax = _y_true.argmax(1)
        _y_true = np.array(torch.zeros(_y_true.shape).scatter(1, _y_true_argmax.unsqueeze(1), 1), dtype=np.int8)
        
        y_true = np.concatenate((y_true, _y_true))
        
        # בניית ה-Predicted Values
        a = _y_pred.argmax(1)
        _y_pred = np.array(torch.zeros(_y_pred.shape).scatter(1, a.unsqueeze(1), 1), dtype=np.int8)

        y_pred = np.concatenate((y_pred, _y_pred))

# --- הדפסת תוצאות ---
print("-" * 30)
print(f'Final Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%')
print("-" * 30)

result = classification_report(y_true, y_pred, digits=5, zero_division='warn')
print(result)

# --- ROC Plot ---
y_true_auc, y_pred_auc = np.array(y_true_auc), np.array(y_pred_auc)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = metrics.roc_curve(y_true_auc, y_pred_auc)
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_true_auc, y_pred_auc)
roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
print('ROC AUC: {:.3f}'.format(roc_auc[1]))
plt.plot(fpr[1], tpr[1], color='darkred', lw=lw, label='ROC curve ({:.3f})'.format(roc_auc[1]))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Meso-4 (Ensemble Soft Voting)')
plt.legend(loc="lower right")

# שמירת הגרף לקובץ
plt.savefig(os.path.join(project_root, 'roc_curve_soft.png'))
print("ROC curve saved.")