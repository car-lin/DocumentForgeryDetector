# Multi-Modal Document Forgery Detection using Deep Learning and GenAI

## Overview
This project presents an end-to-end pipeline for detecting forged documents using a multi-modal deep learning approach. The system classifies documents into three categories:

- Real
- Edited (manually forged)
- AI-generated (GenAI forged)

The model uses multiple feature representations including RGB, ELA, SRM, and FFT to improve detection performance.

---

##  Key Features
- Multi-modal feature fusion (RGB + ELA + SRM + FFT)
- Two-stage classification:
  - Stage 1: Real vs Forged
  - Stage 2: Edited vs AI-generated
- Custom GenAI-based data generation
- Train/Test split (80/20)
- Evaluation using accuracy, confusion matrix, and F1-score
- Gradio-based deployment UI

---

##  Project Structure

```text
project/
│
├── config.yaml
├── requirements.txt
├── README.md
│
├── checkpoints/
│   ├── stage1.pt
│   └── stage2.pt
│
├── data/
│   └── processed/
│       ├── rgb/
|            ├──test/
|            ├──train/
│       ├── ela/
|            ├──test/
|            ├──train/
│       ├── srm/
|            ├──test/
|            ├──train/
│       └── fft/
|            ├──test/
|            ├──train/
│
└── src/
    ├── app.py
    ├── config_utils.py
    ├── dataprep.py
    ├── dataset.py
    ├── evaluate.py
    ├── genAI_forge_class.py
    ├── main.py
    ├── model.py
    ├── predict.py
    └── train.py

```
---

## Prerequisites Installation

pip install -r requirements.txt

---

## Technical Stack

- **Deep Learning**: PyTorch, timm (Vision Transformer), Torchvision  
- **Computer Vision**: OpenCV, Pillow (PIL)  
- **Feature Extraction**:  
  - ELA (Error Level Analysis)  
  - SRM (Spatial Rich Model)  
  - FFT (Frequency Domain Analysis)  

- **Generative AI**: Diffusers, Transformers (HuggingFace), Stable Diffusion (inpainting)  

- **Data Processing**: NumPy, pandas  

- **Evaluation & Metrics**: scikit-learn (confusion matrix, classification report)  

- **Deployment**: Gradio  

- **Configuration & Utilities**: PyYAML  

- **Datasets**:  
    1. **RTM Dataset** – Real and manually edited document images  
    2. **SROIE Dataset** – Real scanned receipts/documents  
    3. **Custom GenAI Dataset** – AI-generated document forgeries created using a diffusion-based inpainting pipeline (stable diffusion)

---

## Usage

### Before You Start
- Update `config.yaml` if needed (paths, dataset locations, checkpoints)
- If you are **only deploying the app**, no changes are required

---

### Prepare Data
```bash
python src/main.py prepare --clean
```

### Train Model

#### Train Stage 1 (Real vs Forged)
```bash
python src/main.py train --stage stage1
```

#### Train Stage 2 (Edited vs AI-generated)
```bash
python src/main.py train --stage stage2
```

#### Train Both Stages (Default)
```bash
python src/main.py train
```

### Evaluate Model
```bash
python src/main.py evaluate
```

### Inference (Single Image)
```bash
python src/main.py infer --image path/to/image.png
```

### Run Complete Pipeline
```bash
python src/main.py all --clean
```

### Deploy Application
```bash
python src/main.py deploy
```

---

## Note

Since this project involves a **large image dataset**, the following steps are computationally intensive and time-consuming:

- AI-generated image creation (GenAI pipeline)  
- Forensic feature extraction (ELA, SRM, FFT)  
- Model training (both stages)  

To simplify usage, **pre-trained checkpoints are provided**, allowing users to directly deploy the application without running the full pipeline.

If you only want to test the system, you can skip data preparation and training and directly run:

```bash
python src/main.py deploy
```
