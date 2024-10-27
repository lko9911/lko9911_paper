# 1. 필요한 라이브러리 설치 및 임포트
import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# 2. CityscapesDataset 클래스 정의
class CityscapesDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform

        # 이미지 및 라벨 파일 목록 생성
        self.image_files = sorted(os.listdir(images_dir))
        self.label_files = sorted([
            f for f in os.listdir(labels_dir)
            if f.endswith('.png')  # color.png만 가져오기
        ])

        # 이미지와 라벨 수 확인
        if len(self.image_files) != len(self.label_files):
            raise ValueError("이미지와 라벨의 수가 일치하지 않습니다.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 이미지 로드
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 라벨 로드
        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        label = cv2.imread(label_path, cv2.IMREAD_COLOR)  # 색상 이미지로 로드

        # 마스크로 변환 (특정 색상 값만 1로 설정, 나머지는 0으로 설정)
        target_color = np.array([128, 64, 128])
        mask = np.all(label == target_color, axis=-1)  # (H, W, 3)에서 특정 색상 검출
        label = np.where(mask, 1, 0).astype(np.uint8)  # target_color에 해당하면 1, 아니면 0

        # 변환 적용
        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']

        return image, label.clone().detach().long()

# 3. 데이터셋 경로 설정
train_images_dir = "content/train/images"
train_labels_dir = "content/train/labels"
val_images_dir = "content/val/images"
val_labels_dir = "content/val/labels"

# 4. 데이터셋 변환 및 데이터 로더 설정
train_transform = A.Compose([
    A.Resize(height=256, width=512),  # 이미지 크기 조정
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),  # albumentations의 ToTensorV2 사용
])
val_transform = train_transform

train_dataset = CityscapesDataset(train_images_dir, train_labels_dir, transform=train_transform)
val_dataset = CityscapesDataset(val_images_dir, val_labels_dir, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

# 5. U-Net 모델 정의
def conv_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model

def conv_trans_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model

def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool

def conv_block_2(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        conv_block(in_dim, out_dim, act_fn),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
    )
    return model

class UnetGenerator(nn.Module):
    def __init__(self, in_dim, out_dim, num_filter):
        super(UnetGenerator, self).__init__()
        act_fn = nn.LeakyReLU(0.2, inplace=True)

        self.down_1 = conv_block_2(in_dim, num_filter, act_fn)
        self.pool_1 = maxpool()
        self.down_2 = conv_block_2(num_filter, num_filter * 2, act_fn)
        self.pool_2 = maxpool()
        self.down_3 = conv_block_2(num_filter * 2, num_filter * 4, act_fn)
        self.pool_3 = maxpool()
        self.down_4 = conv_block_2(num_filter * 4, num_filter * 8, act_fn)
        self.pool_4 = maxpool()

        self.bridge = conv_block_2(num_filter * 8, num_filter * 16, act_fn)

        self.trans_1 = conv_trans_block(num_filter * 16, num_filter * 8, act_fn)
        self.up_1 = conv_block_2(num_filter * 16, num_filter * 8, act_fn)
        self.trans_2 = conv_trans_block(num_filter * 8, num_filter * 4, act_fn)
        self.up_2 = conv_block_2(num_filter * 8, num_filter * 4, act_fn)
        self.trans_3 = conv_trans_block(num_filter * 4, num_filter * 2, act_fn)
        self.up_3 = conv_block_2(num_filter * 4, num_filter * 2, act_fn)
        self.trans_4 = conv_trans_block(num_filter * 2, num_filter, act_fn)
        self.up_4 = conv_block_2(num_filter * 2, num_filter, act_fn)

        self.out = nn.Sequential(
            nn.Conv2d(num_filter, out_dim, kernel_size=3, stride=1, padding=1),
            nn.Softmax(dim=1),  # Softmax 사용
        )

    def forward(self, input):
        down_1 = self.down_1(input)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        bridge = self.bridge(pool_4)

        trans_1 = self.trans_1(bridge)
        concat_1 = torch.cat([trans_1, down_4], dim=1)
        up_1 = self.up_1(concat_1)
        trans_2 = self.trans_2(up_1)
        concat_2 = torch.cat([trans_2, down_3], dim=1)
        up_2 = self.up_2(concat_2)
        trans_3 = self.trans_3(up_2)
        concat_3 = torch.cat([trans_3, down_2], dim=1)
        up_3 = self.up_3(concat_3)
        trans_4 = self.trans_4(up_3)
        concat_4 = torch.cat([trans_4, down_1], dim=1)
        up_4 = self.up_4(concat_4)

        out = self.out(up_4)
        return out

# 6. 모델 초기화
n_classes = 2  # Cityscapes의 클래스 수
in_channels = 3  # 입력 이미지의 채널 수 (RGB)
num_filter = 64  # 필터 수

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UnetGenerator(in_channels, n_classes, num_filter).to(device)

# 7. 손실 함수와 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

import matplotlib.pyplot as plt

def denormalize(image):
    """이미지를 정규화 해제하여 원래 범위로 복원."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0, 1)  # 이미지 값을 0과 1 사이로 클리핑
    return image

def overlay_masks(original, mask, prediction, num_images=3, alpha=0.5):
    plt.figure(figsize=(15, num_images * 5))
    
    for i in range(num_images):
        # 원본 이미지 역정규화
        original_image = denormalize(original[i].permute(1, 2, 0).cpu().numpy())
        
        # 마스크와 예측 마스크 준비
        ground_truth = mask[i].cpu().numpy()
        predicted_mask = torch.argmax(prediction[i], dim=0).cpu().numpy()
        
        # 예측 마스크 오버레이
        overlay_pred = original_image.copy()
        overlay_pred[predicted_mask == 1] = [0, 1, 0]  # 녹색
        
        # Ground Truth 마스크 오버레이
        overlay_truth = original_image.copy()
        overlay_truth[ground_truth == 1] = [1, 0, 0]  # 빨간색
        
        # 투명도 적용
        overlay_pred = cv2.addWeighted(original_image, 1 - alpha, overlay_pred, alpha, 0)
        overlay_truth = cv2.addWeighted(original_image, 1 - alpha, overlay_truth, alpha, 0)

        # 이미지 시각화
        plt.subplot(num_images, 3, i * 3 + 1)
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(num_images, 3, i * 3 + 2)
        plt.imshow(overlay_truth)
        plt.title("Road label Image")
        plt.axis("off")

        plt.subplot(num_images, 3, i * 3 + 3)
        plt.imshow(overlay_pred)
        plt.title("Predicted Image")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# 9. 학습 및 검증 루프
if __name__ == '__main__':
    n_epochs = 20
    best_val_loss = float('inf')

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels.squeeze(1))
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader)
        print(f"train_loss: {avg_train_loss:.4f}")

        # 검증 단계
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels.squeeze(1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"val_loss: {avg_val_loss:.4f}")

        # 매 5 에폭마다 시각화
        if epoch % 5 == 0:
            with torch.no_grad():
                sample_images, sample_labels = next(iter(val_loader))
                sample_images = sample_images.to(device)
                sample_labels = sample_labels.to(device)
                sample_outputs = model(sample_images)
                
                overlay_masks(sample_images, sample_labels, sample_outputs)

        # 가장 낮은 검증 손실을 가진 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_unet_cityscapes_ss.pth")
            print(f"Best model saved with val_loss: {avg_val_loss:.4f}")


