"""Train MobileNetV3 model on custom coordinate dataset to predict target workspace (X, Y) positions."""

import os
import zipfile
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ZIP_NAME = "coord_dataset (5).zip"
DATA_FOLDER = "coord_dataset"

if os.path.exists(ZIP_NAME):
    with zipfile.ZipFile(ZIP_NAME, "r") as zip_ref:
        zip_ref.extractall(".")


class CoordDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        coords = torch.tensor(
            [self.data.iloc[idx, 1], self.data.iloc[idx, 2]], dtype=torch.float32
        )

        if self.transform:
            image = self.transform(image)

        return image, coords


def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])

    dataset = CoordDataset(
        f"{DATA_FOLDER}/labels.csv",
        f"{DATA_FOLDER}/images",
        transform=transform,
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT
    )
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"Training model on {len(dataset)} samples using {device}...")
    model.train()
    for epoch in range(40):
        total_loss = 0.0
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch [{epoch + 1}/40], Loss: {total_loss / len(loader):.4f}"
            )

    torch.save(model.state_dict(), "coord_predictor.pth")
    print("Training complete. Weights saved to 'coord_predictor.pth'.")


if __name__ == "__main__":
    main()