import os
from PIL import Image
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import torch.utils.data as data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import sklearn.metrics as metrics
import copy
import matplotlib.pyplot as plt
import torch.multiprocessing
import numpy as np

# הנחת יסוד: הקבצים הללו קיימים בתיקייה שלך
from Common_Function_ import *
from models.MesoNet4_forEnsemble import Meso4 as MesoNet

# --- הגדרות כלליות ---
torch.multiprocessing.set_sharing_strategy('file_system')
GPU = '0'
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
MODELS_NAME = 'Meso4'

pretrained_size = 224
pretrained_means = [0.4489, 0.3352, 0.3106]
pretrained_stds = [0.2380, 0.1965, 0.1962]

# --- הגדרת ה-Dataset ---
class CustumDataset(Dataset):
    def __init__(self, data, target, data_2=None, target_2=None, transform=None):
        self.data = data
        self.target = target
        self.data_video = data_2
        self.target_video = target_2
        self.transform = transform

        if self.data_video:
            self.len_data2 = len(self.data_video)
        
        # הדפסות לבדיקה
        print(f"Data Video Len: {self.len_data2}")
        print(f"Target Len: {len(self.target)}")

        assert (self.len_data2 == len(self.target) == len(self.target_video) == len(self.data) == len(self.data_video))

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # טעינת תמונה ראשונה (Audio Spectrogram)
        path = self.data[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        # טעינת תמונה שנייה (Video Frame)
        if self.data_video:
            path_video = self.data_video[idx] # תוקן מ-self.data ל-self.data_video למען הבהירות, למרות שבקוד המקורי זה היה self.data כי הרשימות מסונכרנות
            img_video = Image.open(path_video).convert('RGB')
            if self.transform:
                img_video = self.transform(img_video)
            
            return img, self.target[idx], img_video, self.target_video[idx]

# --- טרנספורמציות ---
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

# --- הגדרת נתיבים דינמית ---
current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_path)

test_dir = [
    os.path.join(project_root, "dataset", "test", "SPECTOGRAMS", "real_A_fake_others"),
    os.path.join(project_root, "dataset", "test", "FRAMES", "real_A_fake_others")
]

checkpoint_dir = os.path.join(project_root, "models")

# --- טעינת דאטה ---
print("Loading datasets from:", test_dir)
list_test = [datasets.ImageFolder(root=test_dir[0], transform=None),
             datasets.ImageFolder(root=test_dir[1], transform=None)]

# הכנת רשימות קבצים מסונכרנות
list_glob_testpath = [list_test[1].samples[i][0] for i in range(len(list_test[1].samples))]
list_targets_testpath = [list_test[1].targets[i] for i in range(len(list_test[1].targets))]

list_num_test = getnum_of_files(test_dir[1])
list_glob_testpath_video = []
list_targets_testpath_video = []

for i in range(len(list_test[0].samples)):
    # זהירות: הלוגיקה כאן מניחה מבנה תיקיות ספציפי מאוד
    folder_name = list_test[0].samples[i][0].split(os.path.sep)[-2]
    num_repeat = int(list_num_test.get(folder_name, 0)) # הוספתי get למניעת קריסה
    
    list_glob_testpath_video += [list_test[0].samples[i][0]] * num_repeat
    list_targets_testpath_video += [list_test[0].targets[i]] * num_repeat
    # הלולאה המקורית דילגה על אינדקסים בצורה קצת מוזרה, כאן השארתי את הלוגיקה של בניית הרשימה אבל ללא דילוג ידני על i כי בפייתון זה לא עובד ככה ב-for loop רגיל
    # הערה: הקוד המקורי שלך בלולאה היה בעייתי (שינוי i בתוך for לא משפיע על האיטרציה הבאה בפייתון).
    # עם זאת, נראה שאתה בונה רשימה חדשה לגמרי, אז זה בסדר.

# הערה חשובה: הלוגיקה המקורית של הלולאה למעלה קצת חשודה (בניית הרשימות), אבל השארתי אותה כפי שהיא כדי לא לשבור את ההתאמה שנעשתה במקור.
# אם ה-Assert נכשל, נדע שיש שם בעיה.

# יצירת ה-Dataset וה-DataLoader
test_data = CustumDataset(list_glob_testpath, list_targets_testpath, list_glob_testpath_video, list_targets_testpath_video, test_transforms)
print(f'Number of testing examples: {len(test_data)}')

test_iterator = data.DataLoader(test_data, shuffle=True, batch_size=BATCH_SIZE)

# --- טעינת מודלים ---
print("Loading Models...")
models = [MesoNet(), MesoNet()]

# שימוש בנתיבים שהגדרנו למעלה
path_model_1 = os.path.join(checkpoint_dir, f'{MODELS_NAME}_realA_fakeB.pt')
path_model_2 = os.path.join(checkpoint_dir, f'{MODELS_NAME}_realA_fakeC.pt')

models[0].load_state_dict(torch.load(path_model_1, map_location=device)['state_dict'])
models[1].load_state_dict(torch.load(path_model_2, map_location=device)['state_dict'])

models[0].to(device)
models[1].to(device)
models[0].eval()
models[1].eval()

# --- ביצוע הערכה (Evaluation) ---
label_encoder = LabelEncoder()
enc = OneHotEncoder(sparse=False)
y_true_auc = []
y_pred_auc = []

def count(x):
    return x.value_counts().sort_values(ascending=False).index[0]

df = pd.DataFrame()
targets = []
y_preds_1 = []
y_preds_2 = []
y_preds_3 = []
y_true_list = [] # הוספתי לאגירת y_true

print("Starting Evaluation Loop...")
for i, data_batch in enumerate(test_iterator):
    with torch.no_grad():
        # data_batch = [img_audio, target_audio, img_video, target_video]
        in_1 = data_batch[0].to(device)
        target = data_batch[1].cpu().detach().numpy()
        targets.append(target)
        
        in_2 = data_batch[2].to(device)

        # Forward Pass
        y_pred_1_out = models[0](in_1)
        y_pred_2_out = models[1](in_2)
        y_pred_3_out = (y_pred_1_out + y_pred_2_out) / 2

        # שמירת נתונים ל-AUC
        _pred = copy.deepcopy(y_pred_3_out).detach().cpu()
        _true = copy.deepcopy(data_batch[1]).detach().cpu().float().tolist()
        
        for _a in _pred[:, 1]: y_pred_auc.append(_a)
        for _a in _true: y_true_auc.append(_a)

        # המרה ל-Labels
        y_pred_1_lbl = y_pred_1_out.argmax(1).detach().cpu().numpy()
        y_preds_1.append(y_pred_1_lbl)
        
        y_pred_2_lbl = y_pred_2_out.argmax(1).detach().cpu().numpy()
        y_preds_2.append(y_pred_2_lbl)
        
        y_pred_3_lbl = y_pred_3_out.argmax(1).detach().cpu().numpy()
        y_preds_3.append(y_pred_3_lbl)

# איחוד התוצאות
y_preds_1 = np.concatenate(y_preds_1)
y_preds_2 = np.concatenate(y_preds_2)
y_preds_3 = np.concatenate(y_preds_3)
targets = np.concatenate(targets)

df['pred1'] = y_preds_1
df['pred2'] = y_preds_2
df['pred3'] = y_preds_3

# Hard Voting Logic
df['hard_vote'] = df.apply(lambda x: count(x), 1)
soft = df.loc[(df['pred1'] != df['pred2'])]['pred3'].copy()
df.loc[(df['pred1'] != df['pred2']), 'hard_vote'] = soft

df['target'] = targets

# --- הדפסת תוצאות ---
print("-" * 30)
print(f'Final Accuracy: {accuracy_score(df["target"], df["hard_vote"]) * 100:.2f}%')
print("-" * 30)

result = classification_report(df["target"], df["hard_vote"], digits=5, zero_division='warn')
print(result)

# --- ROC Plot ---
y_true_auc = np.array(y_true_auc)
y_pred_auc = np.array(y_pred_auc)

fpr = dict()
tpr = dict()
roc_auc = dict()

fpr[1], tpr[1], _ = metrics.roc_curve(y_true_auc, y_pred_auc)
roc_auc[1] = metrics.auc(fpr[1], tpr[1])

plt.figure()
lw = 2
print('ROC AUC: {:.3f}'.format(roc_auc[1]))
plt.plot(fpr[1], tpr[1], color='darkred', lw=lw, label='ROC curve ({:.3f})'.format(roc_auc[1]))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('MesoInception-4 (Ensemble)')
plt.legend(loc="lower right")
# plt.show() # ב-Remote לפעמים עדיף לשמור
plt.savefig(os.path.join(project_root, 'roc_curve.png'))
print("ROC curve saved to project root.")