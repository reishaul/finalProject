import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import copy
import time
import os
import torch
from torch.cuda.amp import GradScaler

# --- תיקון ייבוא דינמי ---
try:
    from Multimodal.utils.EarlyStopping import EarlyStopping
    from Multimodal.utils.Common_Function import *
    from Multimodal.models.MesoNet import Meso4
except ImportError:
    # Fallback למקרה שאנחנו מריצים מתוך התיקייה הפנימית
    from utils.EarlyStopping import EarlyStopping
    from utils.Common_Function import *
    from models.MesoNet import Meso4

def TrainMesoNet(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- בדיקת נתיבים משופרת ---
    modes_to_train = []
    
    print("\n--- Checking Data Paths ---")
    if args.path_video and os.path.exists(args.path_video):
        print(f"[V] Video path found: {args.path_video}")
        modes_to_train.append('VIDEO')
    else:
        print(f"[X] Video path NOT found: {args.path_video}")

    if args.path_audio and os.path.exists(args.path_audio):
        print(f"[V] Audio path found: {args.path_audio}")
        modes_to_train.append('AUDIO')
    else:
        print(f"[X] Audio path NOT found: {args.path_audio}")
    
    if not modes_to_train:
        print("Error: No valid data paths found. Exiting training function.")
        return

    print(f"Modes selected for training: {modes_to_train}\n")

    for MODE in modes_to_train:
        train_dir = args.path_video if MODE == 'VIDEO' else args.path_audio
        
        print(f"--> Starting Training for: {MODE}")
        print(f"    Data Directory: {train_dir}")

        EPOCHS = args.epochs
        BATCH_SIZE = args.batch_size
        VALID_RATIO = args.val_ratio
        START_LR = args.lr
        PATIENCE_EARLYSTOP = args.n_early
        SAVE_PATH = args.path_save

        # יצירת תיקיית שמירה אם אינה קיימת
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)

        pretrained_size = 224
        pretrained_means = [0.4489, 0.3352, 0.3106]
        pretrained_stds = [0.2380, 0.1965, 0.1962]
        
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

        # טעינת הדאטה
        try:
            train_data = datasets.ImageFolder(root=train_dir, transform=train_transforms)
        except Exception as e:
            print(f"Error loading ImageFolder from {train_dir}: {e}")
            continue

        n_valid_examples = int(len(train_data) * VALID_RATIO)
        n_train_examples = len(train_data) - n_valid_examples

        train_data, valid_data = data.random_split(train_data, [n_train_examples, n_valid_examples])
        valid_data = copy.deepcopy(valid_data)
        valid_data.dataset.transform = test_transforms

        print(f'    Training examples: {len(train_data)}')
        print(f'    Validation examples: {len(valid_data)}')
        
        train_iterator = data.DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE, num_workers=0) # num_workers=0 למניעת בעיות ב-Windows
        valid_iterator = data.DataLoader(valid_data, shuffle=True, batch_size=BATCH_SIZE, num_workers=0)

        model = Meso4()
        # תמיכה ב-DataParallel רק אם יש יותר מ-GPU אחד
        if torch.cuda.device_count() > 1 and len(args.num_gpu.split(',')) > 1:
             model = nn.DataParallel(model)
        
        model = model.to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        scaler = GradScaler()
        early_stopping = EarlyStopping(patience=PATIENCE_EARLYSTOP, verbose=True)

        optimizer = optim.Adam(model.parameters(), lr=START_LR)
        best_valid_loss = float('inf')
        
        print("    Training loop started...")
        for epoch in range(EPOCHS):
            start_time = time.monotonic()

            train_loss, train_acc_1, train_acc_5 = train(model, train_iterator, optimizer, criterion, scaler, device)
            valid_loss, valid_acc_1, valid_acc_5 = evaluate(model, valid_iterator, criterion, device)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                # שם הקובץ לשמירה
                save_filename = f'{args.model}_{MODE}.pt' # שמות פשוטים יותר
                if hasattr(args, 'model') and args.model == 'MESO4' and MODE == 'VIDEO':
                     # התאמה ספציפית למה שה-Eval מצפה לו, למשל:
                     # אבל כרגע נשמור בשם גנרי, תשנה ידנית אחר כך אם צריך
                     pass

                torch.save({'state_dict': model.state_dict(),
                            'best_acc': valid_acc_1,
                            'val_loss': valid_loss,
                            'epoch': epoch,
                            'lr': START_LR}, 
                            os.path.join(SAVE_PATH, save_filename))

            end_time = time.monotonic()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            print(f'    Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc_1 * 100:6.2f}%')
            print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc_1 * 100:6.2f}%')

            if early_stopping:
                early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("    Early stopping triggered")
                break
        print(f"--> Finished training {MODE}\n")