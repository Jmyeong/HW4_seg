import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# from datasets.dataset import Dataset, Split_Datasets
# from backbones_unet.model.unet import Unet
from backbones_unet.model.losses import DiceLoss
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2
# from model.unet import Unet
from model.DeepLabV3Plus import network

class EarlyStopping:
    def __init__(self, patience=5, delta=0, save_path="checkpoint.pth"):
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.best_val_loss = float('inf')
        self.best_train_loss = float('inf')
        self.best_model_wts = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, train_loss, val_loss, model):
        if val_loss < self.best_val_loss - self.delta:
            self.best_val_loss = val_loss
            self.best_model_wts = model.state_dict() 
            self.counter = 0  
        elif train_loss < self.best_train_loss - self.delta:
            self.best_train_loss = train_loss
            self.counter += 1
            if self.counter >= self.patience:  
                self.early_stop = True

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_wts)

# Split datasets for KFold (For Each patient ID)
class Split_Datasets():
    def __init__(self, path):
        super(Split_Datasets, self).__init__()
        self.image_list = glob(path + "/**/*.tif", recursive=True)
        self.id_list, self.group = self.split(self.image_list)     
        
    def __getitem__(self, index):
        return self.group[index]
    
    def __len__(self):
        return len(self.id_list)
    
    def read_id(self, text):
        parts = text.split("_")
        return parts[-2]

    def split(self, lists):
        ID_list = []
        for text in lists:
            if text.split("_")[-1] != "mask.tif":
                ID = text.split("_")[-2]
                if ID not in ID_list:
                    ID_list.append(ID)
        group = {}
        encoder = LabelEncoder()
        encoded_id = encoder.fit_transform(ID_list)

        for ID in encoded_id:
            group[ID] = []
        for text in lists:
            for key, _ in group.items():
                if self.read_id(text) == encoder.inverse_transform(encoded_id)[key]:
                    group[key].append(text)
        return ID_list, group

class Dataset():
    def __init__(self, path, mode, transform):
        super(Dataset, self).__init__()
        if mode == 'train' or mode == 'valid':
            self.file_list = path  
            self.mask_list = self.make_mask_list(self.file_list)
        else:
            self.test_name = []
            self.file_list = glob(path + "/*.tif", recursive=True)
            for file in self.file_list:
                parts = file.split("/")
                name = parts[-1].split(".")[0] + "_mask.tif"
                self.test_name.append(name)

        self.transform = transform
        self.mode = mode
        
    def __getitem__(self, index):
        image_file_path = self.file_list[index]
        image = tiffile.imread(image_file_path)
        image = Image.fromarray(image)
        image = self.transform(image)
            
        if self.transform and self.mode == 'train':
            gt_file_path = self.mask_list[index]
            gt = tiffile.imread(gt_file_path)
            gt = Image.fromarray(gt)
            gt = self.transform(gt)
            return image, gt
        else:
            return image, self.test_name[index]
    
    def __len__(self):
        return len(self.file_list)

    # Make mask image path lists from input image path lists
    def make_mask_list(self, file_list):
        new_mask_path_list = []
        for file in file_list:
            parts = file.split("_")
            last_part = parts[-1]
            number = last_part.split(".")[0]  
            new_last_part = f"{int(number)}_mask.tif"
            mask_path = ""
            for part in parts[:-1]:
                mask_path = mask_path + part + "_"
            new_mask_path = mask_path + new_last_part
            new_mask_path_list.append(new_mask_path)
        return new_mask_path_list


def dice_coefficient(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 2. * intersection / union if union != 0 else 1.0

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"\nSuccessfully saved in {path}")

def results_to_uint(output, threshold):
    output = (output > threshold).float()
    output = output * 255
    output = output.squeeze(0).cpu().nunmpy().astype(np.uint8).transpose(2, 1, 0)
    return output

def main():
    parser = argparse.ArgumentParser(description='JBNU DL Class HW 4')
    parser.add_argument('--device', type=str, required=False, help='GPU type', default='cuda')
    parser.add_argument('--backbone', type=str, required=False, help='Unet Backbone Type', default='convnext_base')
    parser.add_argument('--pretrained', type=str, required=False, help='Pretrained or not', default=True)
    parser.add_argument('--data_path', type=str, required=False, help='Training data path', default="/ssd1/jm_data/HW4/HW4_data/kaggle_3m")
    parser.add_argument('--n_splits', type=int, required=False, help='KFold implement number', default=7)
    parser.add_argument('--epochs', type=int, required=False, help='Total epoch : n_splits x epochs, default:5x4=20', default=30)
    parser.add_argument('--lr', type=int, required=False, help='learning_rate', default=1e-5)
    parser.add_argument('--save_path', type=str, required=False, help='model save path', default="./checkpoint/model.pth")
    parser.add_argument('--batch_size', type=int, required=False, help='Batch size', default=2)

    args = parser.parse_args()
    
    device = torch.device(args.device)
    batch_size = args.batch_size
    
    summary_save_path = "./summary"
    writer = SummaryWriter(summary_save_path)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    
        # transforms.RandomHorizontalFlip(0.5)
    ])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    def ensemble_eval(model_cls, batch_size, transform):
        print(f"\n\nEvaluating Ensemble...\n\n")
        model_paths = [f"./checkpoint/model_fold{fold+1}.pth" for fold in range(args.n_splits)] # Set model_paths to same with path in training
        models = []
        predictions = []
        
        test_data = Dataset(path="/ssd1/jm_data/HW4/HW4_data/test", mode="test", transform=transform)
        test_loader = DataLoader(test_data, 1, shuffle=False)
        
        for path in model_paths:
            model = model_cls
            model.load_state_dict(torch.load(path))
            model.eval()
            models.append(model)
            
        test_path_lists = []
        
        for i, data in enumerate(tqdm(test_loader)):
            inputs, test_file_path = data
            inputs = inputs.to(device)
            print(inputs.shape)
            fold_preds = []
            test_path_lists.append(test_file_path)
            
            for model in models:
                with torch.no_grad():
                    output = model(inputs)
                    output = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                    fold_preds.append(output)
            
            # Soft Voting
            # Transform to uint8 dtype
            ensemble_pred = np.mean(fold_preds, axis=0) 
            ensemble_pred = (ensemble_pred > 0.5).astype(np.float32)
            ensemble_pred = ensemble_pred * 255
            print(ensemble_pred.shape)
            result = ensemble_pred.astype(np.uint8)
            predictions.append(result)

        # If there is no output dir, then make it
        output_dir = "./outputs"
        os.makedirs(output_dir, exist_ok=True)

        # Save results
        print(f"Saving results...")
        
        for i in range(len(predictions)):
            cv2.imwrite(f"{output_dir}/{test_path_lists[i][0]}", predictions[i].squeeze(-1))
        print(f"Successfully saved!")

    train_data = Split_Datasets(path=args.data_path)
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    avg_valid = 0.0
    
    print(f"\n--------Total epochs : {args.n_splits * args.epochs}--------\n")
    print(f"\n--------Training Start--------\n")
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_data)):
        early_stopping = EarlyStopping(patience=3, delta=0.0, save_path=args.save_path+"/best_model.pth")

        print(f"-----------------------------------------------")
        print(f"Now Fold : {fold + 1}")
        print(f"-----------------------------------------------")

        # model = Unet(
        #     backbone=args.backbone,
        #     in_channels=3,
        #     num_classes=1,
        #     pretrained=args.pretrained
        # ).to(device)
        # model = Unet(3, 1).to(device)
        model = models.segmentation.deeplabv3_resnet101(pretrained=True).to(device)
        model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1)).to(device)

        train_lists = []
        valid_lists = []
                    
        for idx in train_idx:
            train_lists.append(train_data[idx])
        for idx in valid_idx:
            valid_lists.append(train_data[idx])

        # 2D -> 1D
        train_lists = [item for sublist in train_lists for item in sublist]
        valid_lists = [item for sublist in valid_lists for item in sublist]
            
        train_dataset = Dataset(path=train_lists, mode='train', transform=train_transform)
        valid_dataset = Dataset(path=valid_lists, mode='train', transform=valid_transform)

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False, drop_last=True)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, args.lr)
        epochs = args.epochs
        # criterion = DiceLoss()
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(epochs):
            train_loss = 0.0
            model.train()
            for i, data in enumerate(tqdm(train_loader)):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                iters = epoch * len(train_loader) + i
                pred = model(inputs)
                loss = criterion(pred, labels)
                train_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            valid_loss = 0.0
            model.eval()
            dice_avg = 0.0
            with torch.no_grad():
                for i, data in enumerate(tqdm(valid_loader)):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    pred = model(inputs)

                    iters = epoch * len(valid_loader) + i

                    loss = criterion(pred, labels)
                    valid_loss += loss

                    dice_avg += dice_coefficient(pred, labels)

                avg_dice_score = dice_avg / len(valid_loader)
            print(f"epoch : {epoch+1}/{epochs}, Train Loss : {train_loss / len(train_loader)}, Valid Loss : {valid_loss / len(valid_loader)}, Valid Dice Score : {avg_dice_score}")
            
            Apply Early stopping
            early_stopping(train_loss / len(train_loader), valid_loss / len(valid_loader), model)
            if early_stopping.early_stop:
                print(f"Early Stopping triggered!")
                break
        Load Best model
        early_stopping.load_best_model(model)
        
        avg_valid += avg_dice_score
        save_model(model, path=f"./checkpoint/model_fold{fold+1}.pth")
    print(f"\nAvg Valid Dice score : {avg_valid / args.n_splits}")
    ensemble_eval(model_cls=model, batch_size=batch_size, transform=test_transform)
    
if __name__ == "__main__":
    main()
