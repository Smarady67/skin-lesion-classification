# Skin Lesion and Viral Disease Classification using Deep Learning

## Author
**Sokha Marady**  
Batch 13 Software Engineering | Deep Learning Final Project  
Lecturer: Mr. Soklong HIM | Academic Year: 2026 – 2027

## Problem Statement
Automated classification of dermoscopic skin lesions and viral skin diseases is a critical tool for clinical decision support. In regions with limited access to dermatological specialists, a reliable deep learning model can triage patients by flagging potentially malignant lesions (e.g., Melanoma, Basal Cell Carcinoma) or infectious diseases (e.g., Monkeypox, Chickenpox) for urgent review. This project designs, trains, and evaluates deep learning models to classify skin images into 14 distinct diagnostic categories.

## Dataset
**Source:** Skin Lesions Classification Dataset (Kaggle: `ahmedxc4/skin-ds`)  
**License:** CC0: Public Domain  
**Size:** 36,656 images across 14 classes.

The dataset is an aggregation of the ISIC 2019 challenge data (HAM10000, BCN20000, MSK) and additional viral skin disease datasets (Monkeypox, Chickenpox, Measles, Cowpox, HFMD, Healthy). 

### Data Split & Reproducibility
To ensure a strictly fair comparison across all approaches (as required by the project rubric), a fixed train/validation/test split is used. The dataset provides an official pre-defined split, which was converted into a single `data/splits/splits.csv` manifest. This CSV acts as the single source of truth for all PyTorch DataLoaders, guaranteeing that every model is evaluated on the exact same held-out test set with identical test-time preprocessing.

### Class Distribution & Known Bias
The dataset exhibits severe class imbalance, which is a known bias in medical imaging datasets. 
* **Majority Class:** Melanocytic nevi (10,300 training images)
* **Minority Class:** Dermatofibroma (191 training images)
* **Imbalance Ratio:** ~53.9:1

To address this, a **Weighted Cross-Entropy Loss** is applied during training, where class weights are inversely proportional to their frequency in the training set.

## Approaches Compared
In accordance with the project requirements, three distinct deep learning approaches are compared:

1. **Approach 1: Custom CNN from Scratch**  
   A low-capacity baseline model (4 Conv-BN-ReLU-Pool blocks) trained from random initialization. This establishes the inductive bias and capacity floor for the problem.
2. **Approach 2: ResNet-18 (Frozen Backbone / Linear Probe)**  
   Uses ImageNet-pretrained ResNet-18 weights. The convolutional backbone is frozen, and only the final fully connected classification head is trained.
3. **Approach 3: ResNet-18 (Fully Fine-Tuned)**  
   Uses ImageNet-pretrained ResNet-18 weights. The entire network (backbone and head) is unfrozen and fine-tuned end-to-end with a lower learning rate.

## Results Summary

| Approach | Test Accuracy | Test Macro-F1 | Trainable Params | Training Time (min) | Hardware |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1. CNN scratch | 0.5210 | 0.4101 | 99,726 | 158.3 | Tesla T4 |
| 2. ResNet-18 frozen | 0.6184 | 0.5415 | 11,183,694 | 98.0 | Tesla T4 |
| 3. ResNet-18 fine-tuned | 0.8419 | 0.8330 | 11,183,694 | 99.9 | Tesla T4 |

*Note: Approach 3 (Fine-tuned ResNet-18) achieved the highest Macro-F1, demonstrating the effectiveness of transfer learning and fine-tuning on imbalanced medical datasets.*

## Discussion & Error Analysis
While the fine-tuned ResNet-18 achieved a strong overall Test Macro-F1 of 0.83, the classification report reveals a clear pattern in its failures. The model performs exceptionally well on distinct viral and healthy classes (e.g., Monkeypox, HFMD, Healthy, all >0.98 F1). However, it struggles with rare, visually subtle malignant or pre-malignant lesions, specifically Actinic keratoses (F1: 0.57), Squamous cell carcinoma (F1: 0.67), Dermatofibroma (F1: 0.68), and Melanoma (F1: 0.66).  

**Why this happens (Course Concepts):** This is a classic limitation in medical imaging. These specific classes share highly similar textures, colors, and border irregularities. Despite using a Weighted Cross-Entropy Loss to combat the 53.9x class imbalance, the absolute number of training samples for these minority classes remains too low for the model to learn robust, high-capacity decision boundaries. The model defaults to predicting the more common benign classes when uncertain.  

**Limitations & Future Work:** The dataset suffers from multi-source acquisition bias (different resolutions from HAM10000, BCN20000, and MSK). Future work should focus on targeted data augmentation (e.g., CutMix or MixUp) or synthetic data generation (e.g., GANs) specifically for the minority cancer classes to improve clinical safety and per-class recall.

## How to Install and Run

### 1. Prerequisites
* Python 3.10+
* A GPU environment (Google Colab recommended)

### 2. Setup
```bash
# Clone the repository
git clone [YOUR_GITHUB_REPO_URL]
cd skin-lesion-classification

# Install dependencies
pip install -r requirements.txt