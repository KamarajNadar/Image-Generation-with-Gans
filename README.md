# 🚀 Conditional GAN Image Generator

This project implements **Conditional Generative Adversarial Networks (CGANs)** to generate synthetic images based on user input, covering:
- **Handwritten digits** (MNIST dataset)
- **Color object images** (CIFAR-10 dataset)

A simple **Tkinter GUI application** is provided to interactively generate and save images.

---

## 📂 Repository Structure

| File/Folder              | Description                                                      |
|---------------------------|------------------------------------------------------------------|
| `gui_generator.py`        | GUI application to generate images using trained CGAN models.    |
| `train_mnist_cgan.ipynb`   | Jupyter Notebook for training CGAN on the MNIST dataset.         |
| `train_cifar_cgan.ipynb`   | Jupyter Notebook for training CGAN on the CIFAR-10 dataset.      |
| `train_cifar_cgan.ipynb - Colab.html` | Colab-exported HTML for CIFAR-10 training (backup). |

---

## 📦 Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/conditional-gan-image-generator.git
   cd conditional-gan-image-generator
   ```

2. **Install required libraries**:
   ```bash
   pip install torch torchvision numpy matplotlib pillow
   ```

3. **Train models** *(optional, pre-trained models recommended)*:
   - Run `train_mnist_cgan.ipynb` to train the MNIST CGAN.
   - Run `train_cifar_cgan.ipynb` to train the CIFAR-10 CGAN.
   - Save generator models as:
     - `mnist_cgan_generator.pth`
     - `generator_epoch_500.pth`

4. **Run the GUI application**:
   ```bash
   python gui_generator.py
   ```

   ➡️ A window will open where you can input a **digit** (MNIST) or **class name** (CIFAR-10) and generate images.

---

## 🖼️ Supported Classes

- **MNIST Digits:**  
  `0 1 2 3 4 5 6 7 8 9`

- **CIFAR-10 Classes:**  
  `airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`

---

## 📊 Example Outputs

| Dataset | Input | Output Example |
|:-------|:------|:---------------|
| MNIST  | Digit `3` | Handwritten '3' image |
| CIFAR-10 | Class `cat` | Cat-like object image |

*(Insert output images here if needed)*

---

## ⚙️ Model Details

- Generator inputs: 100-dimensional noise + label embedding.
- Generator outputs:
  - 28x28 grayscale image (MNIST)
  - 32x32 RGB image (CIFAR-10)
- Trained with Adam optimizer and Binary Cross-Entropy loss.

---

## 🚀 Future Improvements

- Generate higher resolution images (128x128 or 256x256).
- Enhance CIFAR-10 model stability.
- Deploy GUI as a web application using Streamlit or Flask.

---

## 📚 References

- Ian Goodfellow et al., "Generative Adversarial Nets" (2014)
- PyTorch GAN Tutorials
- MNIST Dataset: [Yann LeCun](http://yann.lecun.com/exdb/mnist/)
- CIFAR-10 Dataset: [CIFAR-10 Official Site](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## 🧑‍💻 Author

- Developed by **KAMARAJ NADAR**
