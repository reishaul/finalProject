import os
from PIL import Image
import pandas as pd
import numpy as np
import copy
import torch.utils.data as data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.multiprocessing
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# הנחת יסוד: הקבצים הללו קיימים בתיקייה שלך
from Common_Function_ import *
# שינוי חשוב: טעינת MesoInception4 כפי שמופיע בקובץ המקורי שלך
from models.MesoNet4_forEnsemble import MesoInception4 as MesoNet

# --- הגדרות חומרה ---
torch.multiprocessing.set_sharing_strategy('file_system')
GPU = '0' # תוקן לכרטיס יחיד
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
MODELS_NAME = 'MesoInception4' # שם המודל שונה כאן בהתאם לקובץ

# טעינת דאטה
print(f"Loading data from: {test_dir}")
list_test = [datasets.ImageFolder(root=test_dir[0], transform=None),
             datasets.ImageFolder(root=test_dir[1], transform=None)]

print(f"Dataset 1 targets: {len(list_test[0].targets)}")
print(f"Dataset 2 targets: {len(list_test[1].targets)}")

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

# חשוב: העברת המודלים ל-GPU (היה חסר בקוד המקורי)
models[0].to(device)
models[1].to(device)
models[0].eval()
models[1].eval()

# --- הרצת הערכה (Evaluation) ---
test_iterator = data.DataLoader(test_data, shuffle=True, batch_size=BATCH_SIZE)

def count(x):
    return x.value_counts().sort_values(ascending=False).index[0]

df = pd.DataFrame()
targets = []
y_preds_1 = []
y_preds_2 = []
y_preds_3 = []

print("Starting Loop...")
for i, batch_data in enumerate(test_iterator):
    with torch.no_grad():
        in_1 = batch_data[0].to(device)
        target = batch_data[1].cpu().detach().numpy()
        targets.append(target)
        
        in_2 = batch_data[2].to(device)
        
        # חיזוי
        y_pred_1_out = models[0](in_1)
        y_pred_2_out = models[1](in_2)
        y_pred_3_out = (y_pred_1_out + y_pred_2_out) / 2
        
        # המרה ל-Labels
        y_pred_1_lbl = y_pred_1_out.argmax(1).detach().cpu().numpy()
        y_preds_1.append(y_pred_1_lbl)
        
        y_pred_2_lbl = y_pred_2_out.argmax(1).detach().cpu().numpy()
        y_preds_2.append(y_pred_2_lbl)
        
        y_pred_3_lbl = y_pred_3_out.argmax(1).detach().cpu().numpy()
        y_preds_3.append(y_pred_3_lbl)

y_preds_1 = np.concatenate(y_preds_1)
y_preds_2 = np.concatenate(y_preds_2)
y_preds_3 = np.concatenate(y_preds_3)

df['pred1'] = y_preds_1
df['pred2'] = y_preds_2
df['pred3'] = y_preds_3

# Hard Voting Logic
df['hard_vote'] = df.apply(lambda x: count(x), 1)
soft = df.loc[(df['pred1'] != df['pred2'])]['pred3'].copy()
df.loc[(df['pred1'] != df['pred2']), 'hard_vote'] = soft

targets = np.concatenate(targets)
print(targets.shape)
print(df.shape)
df['target'] = targets

print("-" * 30)
print(f'Final Accuracy: {accuracy_score(df["target"], df["hard_vote"]) * 100:.2f}%')
print("-" * 30)

# הדפסת ספירות (כמה זויפו וכמה אמיתיים זוהו)
print("Predictions Count:")
print(df["hard_vote"].value_counts())

result = classification_report(df["target"], df["hard_vote"], digits=5, zero_division='warn')
print(result)