import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, random_split, Dataset
from torch.amp import GradScaler
from torch.amp import autocast
import torchvision.transforms.functional as TF
import random


# Choose device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()


# U-net with pretrained encoder (ResNet34)
class UNetSegmentation(nn.Module):

    def __init__(self, num_classes: int):
        super().__init__()
        self.model = smp.Unet(encoder_name="resnet34",encoder_weights="imagenet",in_channels=1,classes=num_classes)

    def forward(self, x):
        return self.model(x)


#  Define dice loss
class DiceLoss(nn.Module):

    def __init__(self, num_classes, eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self,logits,targets):
        probs = torch.softmax(logits,dim=1)
        targets_1hot = torch.nn.functional.one_hot(targets,self.num_classes)
        targets_1hot = targets_1hot.permute(0,3,1,2).float()  # (N, K, H, W)
        dims = (0,2,3)
        intersection = torch.sum(probs*targets_1hot,dims)
        union = torch.sum(probs+targets_1hot,dims)
        dice = (2.0*intersection+self.eps)/(union+self.eps)
        return 1.0-dice.mean()


# Custom pytorch dataset for image augmentation
class SegmentationDataset(Dataset):

    def __init__(self,images:torch.Tensor,labels:torch.Tensor,augment=False):
        self.images = images
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx].clone()
        lbl = self.labels[idx].clone()

        if self.augment:
            # Horizontal flip
            if random.random() > 0.5:
                img = TF.hflip(img)
                lbl = TF.hflip(lbl)

            # Vertical flip
            if random.random() > 0.5:
                img = TF.vflip(img)
                lbl = TF.vflip(lbl)

            # 90-degree rotations (0, 90, 180, 270)
            k = random.randint(0, 3)
            if k > 0:
                img = torch.rot90(img, k, [1, 2])
                lbl = torch.rot90(lbl, k, [0, 1])

        return img, lbl


# scaler = GradScaler(enabled=(device.type == "cuda"))

# Loop for training one epoch
def train_one_epoch(model, loader):
    model.train()
    total_loss = 0.0

    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True).long()
        optimizer.zero_grad()
        logits = model(imgs)
        # print("Logits min/max:", logits.min().item(), logits.max().item())
        # print("Labels min/max:", lbls.min().item(), lbls.max().item())
        loss_ce = ce_loss(logits, lbls)
        loss_dice = dice_loss(logits, lbls)
        loss = loss_ce + 0.5 * loss_dice
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


#  Loop for model evaluation
def evaluate(model, loader, num_classes):
    model.eval()
    total_correct = 0
    total_pixels = 0
    intersection = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True).long()
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == lbls).sum().item()
            total_pixels += lbls.numel()

            for c in range(num_classes):
                p = preds == c
                l = lbls == c
                intersection[c] += (p & l).sum()
                union[c] += (p | l).sum()

    pixel_acc = total_correct / total_pixels
    mean_iou = (intersection / (union + 1e-6)).mean().item()

    return pixel_acc, mean_iou




# Import and adjust data
labels = torch.load(r"D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\k_clipped\k_clipped",map_location="cpu")
labels = labels.squeeze(1)
labels = labels.long()
images = torch.load(r"D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\wrapped_noised_clipped\wrapped_noised_clipped",map_location="cpu")
images = images.float()
images = (images+images.max().item())/(2*images.max().item())

# Get Number of classes
k = labels.max().item()+1

# Adjust initial classes weights
flat_labels = labels.view(-1)
class_counts = torch.bincount(flat_labels,minlength=k).float()
class_weights = 1.0/(class_counts+0.000001)
class_weights = class_weights/class_weights.sum()*k
class_weights = class_weights.to(device)

# Initialize the model
model = UNetSegmentation(num_classes=k)
for p in model.model.encoder.parameters():
    p.requires_grad = False
model = model.to(device)

# Define loss function
ce_loss = nn.CrossEntropyLoss(weight=class_weights)
dice_loss = DiceLoss(k)
optimizer = torch.optim.AdamW(model.parameters(),lr=0.0001,weight_decay=0.0001)

# Divide into training and validation datasets
dataset = SegmentationDataset(images,labels)
train_size = int(0.8*len(dataset))
val_size = len(dataset)-train_size
train_ds, val_ds = random_split(dataset,[train_size, val_size],generator=torch.Generator().manual_seed(42))

# Set up loaders
train_loader = DataLoader(train_ds,batch_size=4,shuffle=True,pin_memory=True)
val_loader = DataLoader(val_ds,batch_size=4,shuffle=False,pin_memory=True)



# Training and validation loop
num_epochs = 40
unfreeze_epoch = 8

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader)
    pixel_acc, mean_iou = evaluate(model, val_loader, k)
    if epoch == unfreeze_epoch:
        for p in model.model.encoder.parameters():
            p.requires_grad = True
        optimizer = torch.optim.AdamW(model.parameters(),lr=0.00001,weight_decay=0.0001)

    print(
        f"Epoch {epoch+1:03d} | "
        f"Train loss: {train_loss:.4f} | "
        f"Pixel Acc: {pixel_acc:.4f} | "
        f"Mean IoU: {mean_iou:.4f}")


# Save the model
torch.save(model.state_dict(), r"D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\models\model_3.pth")













