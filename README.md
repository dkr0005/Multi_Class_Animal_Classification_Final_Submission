## 🐾 ***Project Title:***  
***Multi-Class Animal Image Classification using Transfer Learning (MobileNetV2)***

---

## 📌 ***Project Description:***

This project aims to build an efficient and accurate ***multi-class animal image classifier*** using deep learning and ***transfer learning*** techniques. The main goal is to classify animal images into their respective categories using a pre-trained convolutional neural network.

We use ***MobileNetV2*** as the base model — a lightweight and efficient CNN architecture trained on the ***ImageNet*** dataset. This model is extended with a custom classification head to adapt it for the specific task of animal classification.

By leveraging MobileNetV2's feature extraction capabilities, we reduce training time and improve accuracy, especially with a limited dataset. The project involves preprocessing the image data, augmenting it, training the model using TensorFlow, and evaluating the results using performance metrics like accuracy and classification reports.

---

## 🔍 ***Key Features:***

- ***Transfer Learning with MobileNetV2:***  
  Uses a pre-trained MobileNetV2 model (excluding its top classification layers) to extract useful image features.

- ***Custom Classification Layers:***  
  On top of the MobileNetV2 base, custom layers like `GlobalAveragePooling2D`, `Dense`, `Dropout`, and a `Softmax` output layer are added to enable multi-class classification.

- ***Data Augmentation and Preprocessing:***  
  Uses `ImageDataGenerator` for efficient real-time image augmentation (rescaling, shuffling, and validation split) to improve generalization.

- ***Training Strategy:***  
  The base model layers are frozen initially to retain learned features from ImageNet, and only the custom top layers are trained.

- ***Model Evaluation:***  
  Model performance is evaluated using validation accuracy and a detailed `classification_report` from `sklearn` showing precision, recall, and F1-score for each class.

---

## ⚙️ ***Technologies & Libraries Used:***

- `Python`
- `TensorFlow` and `Keras`
- `NumPy`
- `Matplotlib`
- `scikit-learn`
- `MobileNetV2` (from `tensorflow.keras.applications`)

---

## 🗂️ ***Dataset Overview:***

- The dataset is structured in subdirectories, with each folder representing a specific animal class.
- Images are read and resized to ***224x224 pixels*** as required by the MobileNetV2 input specifications.
- The dataset is divided into ***training (90%)*** and ***validation (10%)*** sets using the `validation_split` parameter in `ImageDataGenerator`.

---

## 🧪 ***Workflow:***

1. ***Load and Visualize Dataset:***  
   - Visualized sample images from different classes using `matplotlib` to understand image structure and size.

2. ***Preprocess Images:***  
   - Rescaled image pixels to the range `[0, 1]`.  
   - Defined image size as `(224, 224)` and used a `batch_size` of `64`.

3. ***Create Image Generators:***  
   - Used `flow_from_directory()` to load images from folders, apply augmentation, and split into training and validation sets.

4. ***Build the Model Architecture:***  
   - Loaded MobileNetV2 with `include_top=False` and froze its layers.  
   - Added the following layers:
     - `GlobalAveragePooling2D` — to reduce dimensionality.
     - `Dense(1024, activation='relu')` — to add non-linearity.
     - `Dropout(0.5)` — to prevent overfitting.
     - `Dense(num_classes, activation='softmax')` — final output layer.

5. ***Compile the Model:***  
   - Optimizer: `Adam(learning_rate=0.0001)`  
   - Loss Function: `categorical_crossentropy`  
   - Evaluation Metric: `accuracy`

6. ***Train the Model:***  
   - Trained on the training set and validated on the validation set using `.fit()` or `.fit_generator()` (based on implementation).

7. ***Evaluate the Model:***  
   - Generated classification metrics such as precision, recall, F1-score using `classification_report` from `sklearn.metrics`.

---

## 📊 ***Model Summary:***

- The model summary includes all the layers, trainable parameters, and architecture, showing how MobileNetV2 is extended for multi-class classification.

---

## 🎯 ***Conclusion:***

This project successfully demonstrates how ***transfer learning*** can be used to develop a powerful image classification system with limited data and compute resources. MobileNetV2 provides an ideal balance between performance and efficiency.

By using a pre-trained model, extensive preprocessing, and a custom top-layer design, the classifier achieves high accuracy and can be easily adapted to other multi-class image classification problems.

---
