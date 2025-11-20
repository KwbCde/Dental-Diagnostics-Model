[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
---

# **Dental Imaging Model EfficientNet-B0 Classifier + Grad-CAM**

A dental imaging classifier built with **PyTorch**, **EfficientNet-B0** and **Grad-CAM**.
The model predicts four categories from oral photos:

* **Healthy**
* **Plaque**
* **Gum Inflammation**
* **Unknown**

The project includes:

* A **training pipeline** (dataset -> transforms -> model -> metrics)
* A **inference engine**
* **GradCAM explainability**
* A **Streamlit UI** for visualization

---

## **Features**

### **Model**

* Backbone: **EfficientNet-B0 (ImageNet-pretrained)**
* Custom classifier head (4 classes)
* Finetuned with:

  * Cross entropy loss
  * Adam optimizer
  * StepLR scheduler
  * Gradient clipping
  * Image augmentations
* Device aware (CUDA -> MPS -> CPU)

### **Training Notebook**

The notebook includes:

1. Imports
2. Config
3. Dataset & transforms
4. DataLoaders
5. Model setup
6. Loss, optimizer, scheduler
7. Training loop (with tqdm)
8. Validation
9. Loss/Accuracy curves
10. Confusion matrix
11. Precision/Recall/F1
12. Grad-CAM visualizations
13. Model saving

gives reproducible training pipeline.

### **Inference Engine**

Located in `src/inference.py`, supports:

* Single image prediction
* Softmax probabilities
* Grad-CAM heatmaps for interpretability

### **Streamlit UI (`ui/app.py`)**

* Upload an image
* See prediction and confidence
* Visualize Grad-CAM:

  * Original
  * Heatmap
  * Overlay
* Device displayed automatically

---

## **Project Structure**

```
Dental-Diagnostics-Model/
│
├── ml/
│   ├── data/                # not included (placeholder README only)
│   ├── models/
|   |   └── dental_classifier.pth      
│   └── notebooks/
│       └── 01_training_pipeline.ipynb
│
├── src/
│   ├── config.py
│   ├── gradcam.py
│   ├── inference.py
│   ├── model_loader.py
│   ├── utils.py
│   └── __init__.py
│
├── ui/
│   └── app.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## **Installation**

```bash
git clone https://github.com/KwbCde/Dental-Diagnostics-Model.git
cd Dental-Diagnostics-Model
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## **Running Inference**

Place trained weights here:

```
ml/models/dental_classifier.pth
```
Dont forget to delete the old one

Then:

```python
from src.inference import InferenceEngine

engine = InferenceEngine("ml/models/dental_classifier.pth", device="cuda")
result = engine.predict_with_gradcam("path/to/image.jpg")

print(result["class_name"], result["confidence"])
```

---

## **Running the Streamlit App**

```bash
streamlit run ui/app.py
```

Upload an image -> get prediction and Grad-CAM visualization.

---

## **Training**

The full training pipeline is inside:

```
ml/notebooks/01_training_pipeline.ipynb
```

It covers data loading, augmentations, training loops, validation, metrics and explainability.

---

## **Model Weights & Data**

* **Weights are included in the repo** (see `ml/model`).
* **Data is not included**

To train:

* Prepare a dataset:

  ```
  data/
    train/
      gum_inflammation/
      healthy/
      plaque/
      unknown/
    val/
      gum_inflammation/
      healthy/
      plaque/
      unknown/
  ```

---

## **Tech Stack**

* PyTorch
* TorchVision
* scikit-learn
* Streamlit
* OpenCV
* Matplotlib / Seaborn
* tqdm

---

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

