import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import os

# Define a simple denoising autoencoder
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2, padding=0),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Define the dataset and dataloader
class FaceDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Adjust size based on model
    transforms.ToTensor(),
])

dataset = FaceDataset("Humans", transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize model, loss function, and optimizer
model = DenoisingAutoencoder().to("cpu")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4)

num_epochs = 5

for epoch in range(num_epochs):
    for images in tqdm(dataloader):
        noisy_images = images + 0.1 * torch.randn_like(images)  # Add noise
        
        # Forward pass
        outputs = model(noisy_images)
        
        # Compute loss
        loss = criterion(outputs, images)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs} completed. Loss: {loss.item()}")

# Save the model
#torch.save(model.state_dict(), "path/to/save/denoising-autoencoder.pth")
