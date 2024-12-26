import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from diffusers import StableDiffusionPipeline
from torch.optim import AdamW
import torch.nn.functional as F
from tqdm.auto import tqdm

# Define the dataset
class CustomImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGBA")  # Ensure image is in RGB format
        if self.transform:
            image = self.transform(image)
        return image

# Define transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]),  # Normalize for 3 channels (RGB)
])

# Load dataset
dataset = CustomImageDataset("Humans", transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Load Stable Diffusion pipeline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipeline.to(device)

# Prepare optimizer
optimizer = AdamW(pipeline.unet.parameters(), lr=5e-6)
pipeline.unet.train()  # Set the model to training mode

# Fine-tuning loop
default_caption = "A photo of a human face"  # Default caption for all images

for epoch in range(5):  # Number of epochs
    for images in tqdm(dataloader, desc=f"Epoch {epoch + 1}/5"):
        images = images.to(device)

        # Generate random noise
        noise = torch.randn_like(images).to(device)
        timesteps = torch.randint(0, 1000, (images.size(0),), device=device).long()

        # Add noise to the images
        noisy_images = pipeline.scheduler.add_noise(images, noise, timesteps)

        # Tokenize the default caption
        text_inputs = pipeline.tokenizer(default_caption, return_tensors="pt").to(device)

        # Encode text to get encoder_hidden_states
        with torch.no_grad():
            encoder_hidden_states = pipeline.text_encoder(text_inputs.input_ids)[0]

        # Forward pass through UNet
        noise_pred = pipeline.unet(noisy_images, timesteps, encoder_hidden_states=encoder_hidden_states).sample

        # Compute loss
        loss = F.mse_loss(noise_pred, noise)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item()}")
