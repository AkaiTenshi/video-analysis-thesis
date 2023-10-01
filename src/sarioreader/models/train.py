import os
import tarfile

import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from sarioreader.models.model import CRNN_Seq
from sarioreader.tools.kit import download_with_progressbar

# Download and extract SVHN dataset (original format)
url = "http://ufldl.stanford.edu/housenumbers/train.tar.gz"
filename = "train.tar.gz"

if not os.path.exists(filename):
    download_with_progressbar(url, filename)

with tarfile.open(filename, "r:gz") as tar:
    tar.extractall()


# Load the dataset
class SVHNDataset(Dataset):
    def __init__(self, mat_file, transform=None):
        self.data = h5py.File(mat_file, "r")
        self.transform = transform

    def __len__(self):
        return len(self.data["digitStruct"]["name"])

    def __getitem__(self, idx):
        name = self.data["digitStruct"]["name"][idx][0]
        bbox = self.data["digitStruct"]["bbox"][idx].item()

        # Extract the label
        label = []
        for i in range(len(self.data[bbox]["label"])):
            label.append(int(self.data[bbox]["label"][i][0]))

        # Load the image
        img = Image.open(
            os.path.join("train", self.data[name][()].tobytes().decode())
        )

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label)


transform = transforms.Compose(
    [
        transforms.Resize((32, 100)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

dataset = SVHNDataset("train/digitStruct.mat", transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the model, loss, and optimizer
num_classes = 10  # Digits: 0-9
model = CRNN_Seq(num_classes)
ctc_loss = nn.CTCLoss(blank=10)  # 10 will act as the blank label
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
num_epochs = 10
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)

        input_lengths = torch.full(
            (labels.shape[0],), outputs.shape[1], dtype=torch.long
        )
        target_lengths = torch.tensor([len(label) for label in labels])

        concatenated_labels = torch.cat(
            labels
        )  # Concatenate all labels in a batch
        loss = ctc_loss(
            outputs, concatenated_labels, input_lengths, target_lengths
        )
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}"  # noqa: E501
            )

# Save the model
torch.save(model.state_dict(), "crnn_seq_svhn_model.pth")
