# Lung Disease Classification

![](https://github.com/SuhasR3/Lung-Disease-Classification/blob/main/Datasets/lung_banner.jpg)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

This project, *Performative Analysis of Optimizers on Fine-Tuned Lung Disease Classification*, evaluates how different optimizers affect the performance of image classification models in detecting lung conditions such as bacterial pneumonia, COVID-19, tuberculosis, viral pneumonia, and normal lungs using chest X-ray images. 

---

### Dataset
- **Source**: Publicly available chest X-ray repositories
- **Size**: 10,000 augmented images
- **Classes**: Normal, Bacterial Pneumonia, COVID-19, Tuberculosis, Viral Pneumonia
- **Preprocessing**: Resized to 224x224x3, normalized, augmented (flipping, rotation)

### Models
1. **ZF-Net**: A CNN optimized for feature extraction with convolutional and fully connected layers.
2. **Sequencer2d**: An LSTM-based model replacing self-attention with bi-directional LSTM blocks.
3. **ResNet-50**: A pre-trained residual neural network with 50 layers, fine-tuned for classification.

### Optimizers
- **SGD**: Stochastic Gradient Descent with Nesterov momentum (0.9).
- **AdamW**: Adaptive optimizer with weight decay, default betas (0.9, 0.999).
- **Adadelta**: Adaptive learning rate optimizer with weight decay (1e-2).

---
## Results

### Key Findings
- **AdamW**: Excels in baseline settings with smaller batch sizes and learning rates.
- **SGD**: Performs best with larger batch sizes, showing strong generalization.
- **Adadelta**: Struggles to converge consistently across configurations.

### Metrics
- Training Loss
- Validation Loss
- Weighted F1 Score

Loss curves and F1 scores are logged during training. Refer to `SML Project Final report/` for outputs.

---

## Installation

### Prerequisites
- Python 3.8+
- PyTorch
- Hugging Face Transformers (for Sequencer2d)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/SuhasR3/Lung-Disease-Classification.git
   cd Lung-Disease-Classification
   ```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Download the dataset and place it in the Dataset/ directory.

## Usage

### Training Models
Run the training script with your desired configuration:
```bash
python train.py --model zfnet --optimizer adamw --batch-size 32 --lr 1e-4 --epochs 50
```
Options: --model [zfnet, sequencer2d, resnet50], --optimizer [sgd, adamw, adadelta]

### Hyperparameters
- Batch Size: 16, 32, 64
- Learning Rate: 1e-2, 1e-4
- Dropout (ZF-Net): 0.25, 0.5


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Conducted as part of Group 15 at ASU.
- Trained on the SOL supercomputer.
- Inspired by research in optimizer performance and medical imaging.
