# AI-Algorithms-to-Detect-Explosives-in-Waste-Receptacles
Development of AI Algorithms to Detect Explosives in Waste Receptacles Using Vision Transformers and Swin Transformers

## Overview
This project develops AI-based image classification algorithms to detect explosives hidden in waste receptacles. The work compares **Vision Transformer (ViT)** and **Swin Transformer** models for both:

- **Binary classification**: IED vs. real waste
- **Multi-class classification**: IED, glass, paper, cardboard, plastic, metal, and trash

The models were trained using the **I2Net** and **TrashNet** datasets, and the report concludes that the **Swin Transformer** performed better overall than the Vision Transformer. fileciteturn1file0L113-L123 fileciteturn1file0L127-L137 fileciteturn1file0L139-L141

## Objective
The goal of this project is to build AI algorithms for detecting explosives in waste bins using Transformer-based computer vision models.

## Key Features
- Binary waste classification: **IED vs real waste**
- Multi-class waste classification across **7 categories**
- Transfer learning with pre-trained Transformer models
- Evaluation using accuracy, precision, recall, and F1-score
- Confusion-matrix-based performance analysis
- Comparison between ViT and Swin Transformer models

## Datasets Used
The project uses two public datasets:

- **[I2Net](https://github.com/jessieAmoakoh/I2Net)** for IED images
- **[TrashNet](https://www.kaggle.com/datasets/feyzazkefe/trashnet)** for non-IED waste images

The report states that the I2Net dataset was split into training, validation, and test sets, and TrashNet includes categories such as glass, paper, cardboard, plastic, metal, and trash.

## Models
### Vision Transformer
- Pre-trained model: `vit_base_patch16_224`
- Used for both binary and multi-class classification

### Swin Transformer
- Pre-trained model: `swin_base_patch4_window7_224`
- Used for both binary and multi-class classification

## Experimental Setup
- Images resized to **224 × 224**
- Training done on **CPU**
- Optimizer: **AdamW**
- Loss function: **CrossEntropyLoss**
- Transfer learning used with pre-trained weights
- Evaluation metrics: **Accuracy, Precision, Recall, F1 Score**

Defined hyperparameters in the report include:
- Batch size: `16`
- Learning rate: `1e-4`
- Epochs:
  - ViT binary: `30`
  - ViT multi-class: `10`
  - Swin binary: `10`
  - Swin multi-class: `10` 

## Results
### Vision Transformer
- **Binary classification**
  - Accuracy: `0.9984`
  - Precision: `0.9986`
  - Recall: `0.9984`
  - F1 Score: `0.9985`

- **Multi-class classification**
  - Accuracy: `0.9197`
  - Precision: `0.9263`
  - Recall: `0.9197`
  - F1 Score: `0.9209`

### Swin Transformer
- **Binary classification**
  - Accuracy: `1.0000`
  - Precision: `1.0000`
  - Recall: `1.0000`
  - F1 Score: `1.0000`

- **Multi-class classification**
  - Accuracy: `0.9855`
  - Precision: `0.9856`
  - Recall: `0.9855`
  - F1 Score: `0.9855`

## Conclusion
We found that the **Swin Transformer** is the best model for detecting explosives hidden in waste bins, because it achieved higher evaluation metrics than the Vision Transformer in both binary and multi-class classification tasks.

## Repository Structure
```text
.
├── dataset/            # Links to Dataset files
├── notebooks/          # Training / experiment notebooks
├── models/             # Links to Saved model checkpoints
└── README.md
```

## Report
[View](Group2_Report_Ass2.pdf)

## Requirements
Typical Python dependencies for this project include:
- Python
- PyTorch
- timm
- torchvision
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- Pillow

## How to Run
1. Clone the repository.
2. Install the required dependencies.
3. Prepare the I2Net and TrashNet datasets in the expected folder structure.
4. Run the training script for the chosen model.
5. Evaluate the saved model on the test set.
6. Review the confusion matrix and classification metrics.

## Contributors
- **Fathima Shamra**
- **Nisini Silva**
