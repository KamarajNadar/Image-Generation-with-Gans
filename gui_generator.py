import torch
import torch.nn as nn
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import torchvision.transforms as transforms
import os
import time
from tkinter import messagebox

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constants
NOISE_DIM = 100
MNIST_CLASSES = [str(i) for i in range(10)]
CIFAR_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Define Generators
class MNISTGenerator(nn.Module):
    def __init__(self, noise_dim=100, num_classes=10):
        super(MNISTGenerator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(noise_dim + num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        labels = self.label_emb(labels)
        input = torch.cat((noise, labels), -1)
        return self.model(input).view(-1, 1, 28, 28)

class CIFARGenerator(nn.Module):
    def __init__(self, noise_dim=100, num_classes=10):
        super(CIFARGenerator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(noise_dim + num_classes, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 3 * 32 * 32),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        labels = self.label_emb(labels)
        x = torch.cat([noise, labels], dim=1)
        x = self.model(x)
        x = x.view(x.size(0), 3, 32, 32)
        return x


# Load the Generators
mnist_generator = MNISTGenerator().to(device)
mnist_generator.load_state_dict(torch.load('mnist_cgan_generator.pth', map_location=device))
mnist_generator.eval()

cifar_generator = CIFARGenerator().to(device)
cifar_generator.load_state_dict(torch.load('generator_epoch_500.pth', map_location=device))
cifar_generator.eval()

# Helper Functions
def generate_images(input_text, dataset):
    if dataset == "MNIST":
        digits = list(input_text.strip())
        if not all(d.isdigit() and int(d) in range(10) for d in digits):
            messagebox.showerror("Error", "Invalid MNIST input! Enter only digits 0-9 without space.")
            return None

        noise = torch.randn(len(digits), NOISE_DIM, device=device)
        labels = torch.tensor([int(d) for d in digits], device=device)
        with torch.no_grad():
            fake_imgs = mnist_generator(noise, labels).cpu()

        images = []
        for img in fake_imgs:
            img = img.squeeze().detach().numpy()
            img = (img + 1) / 2.0  # Denormalize
            img_pil = Image.fromarray((img * 255).astype('uint8'), mode='L').resize((64, 64))
            img_pil = img_pil.convert("RGB")  # Convert MNIST to RGB
            images.append(img_pil)

        return merge_images(images)

    elif dataset == "CIFAR-10":
        classes = input_text.strip().split()
        if len(classes) != 1 or classes[0].lower() not in CIFAR_CLASSES:
            messagebox.showerror("Error", "Invalid CIFAR-10 input! Enter exactly one valid class name.")
            return None

        noise = torch.randn(1, NOISE_DIM, device=device)
        label = torch.tensor([CIFAR_CLASSES.index(classes[0].lower())], device=device)
        with torch.no_grad():
            fake_img = cifar_generator(noise, label).cpu()

        img = (fake_img.squeeze(0) + 1) / 2.0  # Denormalize
        img_pil = transforms.ToPILImage()(img).resize((128, 128))
        return img_pil

    else:
        messagebox.showerror("Error", "Select a valid dataset.")
        return None

def merge_images(images):
    if not images:
        return None

    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    merged_img = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    x_offset = 0
    for img in images:
        merged_img.paste(img, (x_offset, (max_height - img.height) // 2))
        x_offset += img.width

    return merged_img

def on_generate():
    user_input = entry.get().strip()
    dataset = dataset_var.get()

    if not user_input:
        messagebox.showerror("Error", "Input cannot be empty!")
        return

    img = generate_images(user_input, dataset)
    if img:
        img = img.resize((min(500, img.width), img.height))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

        root.geometry(f"{max(600, img.width + 100)}x600")

def on_save():
    if hasattr(image_label, 'image'):
        # Make sure the folder exists
        save_dir = os.path.join(os.getcwd(), "generated_images")
        os.makedirs(save_dir, exist_ok=True)
        
        # Create a unique filename using timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        img_path = os.path.join(save_dir, f"generated_image_{timestamp}.png")
        
        # Save the image
        image_label.image._PhotoImage__photo.write(img_path, format='png')
        
        messagebox.showinfo("Saved", f"Image saved at {img_path}!")

# GUI
root = tk.Tk()
root.title("Image Generation with GANs")
root.geometry("500x300")
root.configure(bg="white")

# Entry field
entry = tk.Entry(root, font=("Arial", 14), width=20)
entry.grid(row=0, column=0, padx=10, pady=10)

# Dataset dropdown
dataset_var = tk.StringVar(value="MNIST")
dataset_combo = ttk.Combobox(root, textvariable=dataset_var, values=["MNIST", "CIFAR-10"], state="readonly", width=10)
dataset_combo.grid(row=0, column=1, padx=5)

# Generate button
generate_button = tk.Button(root, text="Generate", command=on_generate, font=("Arial", 12), bg="blue", fg="white")
generate_button.grid(row=0, column=2, padx=5)

# Save button
save_button = tk.Button(root, text="Save Image", command=on_save, font=("Arial", 12), bg="blue", fg="white")
save_button.grid(row=1, column=0, columnspan=3, pady=5)

# Image label
image_label = tk.Label(root, bg="white")
image_label.grid(row=2, column=0, columnspan=3)

# Info label
info_text = (
    "Classes Available:\n"
    "🔵 MNIST: 0 1 2 3 4 5 6 7 8 9\n"
    "🟢 CIFAR-10: airplane automobile bird cat deer dog frog horse ship truck\n\n"
    "➡️ MNIST: Enter digits without space (e.g., 1234)\n"
    "➡️ CIFAR-10: Enter exactly one class name (e.g., cat)"
)
info_label = tk.Label(root, text=info_text, font=("Arial", 10), fg="black", bg="white", justify="left", anchor="w")
info_label.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky="w")
root.mainloop()
