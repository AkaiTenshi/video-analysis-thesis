import os
import tarfile
import urllib.request

import h5py
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from sarioreader.models.model import CRNN_Seq

# Download the test SVHN dataset (original format)
url_test = "http://ufldl.stanford.edu/housenumbers/test.tar.gz"
filename_test = "test.tar.gz"

if not os.path.exists(filename_test):
    urllib.request.urlretrieve(url_test, filename_test)

with tarfile.open(filename_test, "r:gz") as tar:
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
            os.path.join("test", self.data[name][()].tobytes().decode())
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

test_dataset = SVHNDataset("test/digitStruct.mat", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the model
num_classes = 10  # Digits: 0-9
model = CRNN_Seq(num_classes)
model.load_state_dict(torch.load("crnn_seq_svhn_model.pth"))
model.eval()


def decode(outputs):
    _, predicted_indices = outputs.max(2)
    sequences = []
    for idx_seq in predicted_indices:
        seq = [
            str(idx) for idx in idx_seq if idx != 10
        ]  # 10 is the blank label
        sequences.append("".join(seq))
    return sequences


# Evaluate the model
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predicted_sequences = decode(outputs)
        label_sequences = [
            "".join([str(digit.item()) for digit in label]) for label in labels
        ]

        total += len(label_sequences)
        correct += sum(
            [
                pred == true
                for pred, true in zip(predicted_sequences, label_sequences)
            ]
        )

print(f"Accuracy: {100 * correct / total:.2f}%")
