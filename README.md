# WordForThought
A collection of four supervised machine learning projects covering regression, binary classification, multi-class image classification, and speech emotion recognition using Python (Pandas, Scikit-learn, TensorFlow/Keras, Librosa).

---

## Projects Overview

| # | Notebook | Task | Dataset | Best Result |
|---|----------|------|---------|-------------|
| 1 | `01_Linear_Regression_Project.ipynb` | Regression | Ecommerce Customers (500 rows) | R² = 0.989 |
| 2 | `02_Logistic_Regression_Project.ipynb` | Binary Classification | Advertising Clicks (1,000 rows) | Accuracy = 93% |
| 3 | `CNN_on_Animals.ipynb` | Image Classification (15 classes) | Animal Images (1,646 images) | Test Accuracy ≈ 70.45% |
| 4 | `Librosa_CNN.ipynb` | Speech Emotion Recognition (6 classes) | RAVDESS + CREMA-D (8,498 audio files) | Test Accuracy ≈ 54.88% |

---

## 1. Linear Regression — Ecommerce Customer Spend Prediction

### Objective
Predict the **yearly amount spent** by e-commerce customers and advise the company on whether to invest in their mobile app or website experience.

### Dataset
- **500 customer records** (no missing values)
- Features: `Avg. Session Length`, `Time on App`, `Time on Website`, `Length of Membership`
- Target: `Yearly Amount Spent` (continuous; mean ≈ $499.31, std ≈ $79.31)

### Approach
1. **EDA** — Seaborn joint plots (Time on App vs Spend, Time on Website vs Spend, hex plots) and pairplot to explore feature correlations.
2. **Feature Selection** — All 4 numerical columns used as predictors.
3. **Train/Test Split** — 70% / 30% (`random_state=101`).
4. **Model** — `sklearn.linear_model.LinearRegression`.
5. **Evaluation** — MSE, MAE, RMSE, R² score; residual histogram to verify normality.

### Results

| Metric | Value |
|--------|-------|
| R² Score | **0.9890** |
| MAE | 7.23 |
| MSE | 79.81 |
| RMSE | 8.93 |

**Learned Coefficients:**

| Feature | Coefficient ($/unit increase) |
|---------|-------------------------------|
| Avg. Session Length | +25.98 |
| Time on App | **+38.59** |
| Time on Website | +0.19 |
| Length of Membership | **+61.28** |

**Conclusion:** `Length of Membership` is the strongest revenue driver (+$61.28/yr per unit). `Time on App` contributes ~203× more than `Time on Website` per unit, indicating the app significantly outperforms the website. The decision to develop app vs. website depends on company strategy — strengthen the existing advantage (app) or close the performance gap (website).

---

## 2. Logistic Regression — Ad Click Prediction

### Objective
Predict whether an internet user will **click on an advertisement** based on behavioral and demographic features.

### Dataset
- **1,000 user records** (no missing values)
- Features used: `Daily Time Spent on Site`, `Age`, `Area Income`, `Daily Internet Usage`, `Male`
- Target: `Clicked on Ad` (binary: 0 = did not click, 1 = clicked)

### Approach
1. **EDA** — Age histogram, joint plots (Age vs Area Income, Age vs Daily Time Spent, Time on Site vs Daily Internet Usage), pairplot colored by `Clicked on Ad`.
2. **Train/Test Split** — 70% / 30% (`random_state=101`).
3. **Model** — `sklearn.linear_model.LogisticRegression` (L-BFGS solver).
4. **Evaluation** — Classification report: precision, recall, F1-score per class.

### Results

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (No Click) | 0.91 | 0.95 | 0.93 | 157 |
| 1 (Clicked) | 0.94 | 0.90 | 0.92 | 143 |
| **Overall Accuracy** | | | **0.93** | **300** |

**Conclusion:** The logistic regression model achieves **93% accuracy** with balanced precision and recall on both classes. Simple behavioral features — time on site, daily internet usage, age, and area income — prove to be strong predictors of ad-click behaviour.

---

## 3. CNN — Animal Image Classification

### Objective
Classify wildlife images into **15 animal categories** using a Convolutional Neural Network.

### Dataset
- **1,646 JPEG images** across 15 classes:
  `Bear, Bird, Cat, Cow, Deer, Dog, Dolphin, Elephant, Giraffe, Horse, Kangaroo, Lion, Panda, Tiger, Zebra`
- Source: Google Drive (`animal_data.zip`)
- Preprocessing: Resized to **96 × 96 × 3**, pixel values normalized to `[0, 1]` (float32).
- Labels: Integer encoded via `sklearn.LabelEncoder`, then one-hot encoded (`to_categorical`, 15 classes).

### Approach
1. **Data Loading** — Image paths collected with `glob`, loaded and stacked into a NumPy array.
2. **Preprocessing** — Resize + normalize (float32, range 0–1).
3. **Split** — 80% train / 16% test / 4% validation (`random_state=102`, stratified shuffle).
4. **Model Architecture** (Keras Sequential CNN):

| Layer | Config |
|-------|--------|
| Conv2D | 16 filters, 3×3, ReLU, `padding=same`, input (96,96,3) |
| MaxPooling2D | 2×2 |
| Conv2D | 32 filters, 3×3, ReLU, `padding=same` |
| MaxPooling2D | 2×2 |
| Dropout | 0.25 |
| Flatten | — |
| Dense | 64 units, ReLU |
| Dropout | 0.50 |
| Dense (output) | 15 units, Softmax |

- **Total parameters:** 1,185,775 (4.52 MB)
- **Optimizer:** Adam | **Loss:** Categorical Cross-Entropy | **Epochs:** 50

### Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **70.45%** |
| Test Loss | 2.1486 |

- Training accuracy grew from ~9.9% (epoch 1) to ~70%+ over 50 epochs, with validation tracking closely.
- Sample prediction on 5 random test images: 4/5 correctly classified (Dog ✓, Giraffe ✓, Cat ✗→Giraffe, Cow ✓, Dog ✓).

**Conclusion:** The lightweight 2-block CNN achieves ~70% accuracy on a 15-class problem with only 1,646 images. Performance is constrained by dataset size; transfer learning (MobileNet, ResNet) or data augmentation would likely push accuracy significantly higher.

---

## 4. Librosa CNN — Speech Emotion Recognition

### Objective
Recognize **human emotion from speech audio** by extracting audio features (Mel Spectrograms, MFCCs, ZCR, RMS Energy) and classifying them with a deep CNN into **6 emotion categories**.

### Dataset
Two publicly available datasets combined:

| Dataset | Source | Notes |
|---------|--------|-------|
| **RAVDESS** | Ryerson Audio-Visual Database of Emotional Speech and Song | 24 actors, 8 emotions originally |
| **CREMA-D** | Crowd-sourced Emotional Multimodal Actors Dataset | `.wav` files, filename-encoded labels |

- **Combined total (raw):** 8,882 audio files across 8 emotions.
- **After filtering** (`calm` and `surprise` removed due to very low sample counts):
  - **8,498 audio files**, **6 emotions**: `angry, disgust, fear, happy, neutral, sad`
  - Each class has ~1,463 samples (except neutral: 1,183).

### Approach

#### 1. EDA & Visualization
Per-emotion sample visualizations:
- **Mel Spectrograms** — frequency content over time.
- **MFCCs** (13 coefficients) — spectral envelope representation.
- **Waveforms** — amplitude vs. time.
- **RMS Energy** — overall loudness over time.
- **Zero Crossing Rate (ZCR)** — signal sign-change rate.

#### 2. Feature Extraction (per audio file, `sr=None`)
All features extracted with `librosa`, padded/truncated to a fixed `MAX_LEN=130` frames:

| Feature | Dimensions | Description |
|---------|------------|-------------|
| Mel Spectrogram | (130, 64) | 64 mel bands, power converted to dB |
| ZCR | (130, 1) | Zero Crossing Rate per frame |
| RMS Energy | (130, 1) | Root Mean Square Energy per frame |
| MFCC | (130, 13) | 13 Mel-Frequency Cepstral Coefficients |

Features concatenated along the last axis → shape per sample: **(130, 79)**
- Final dataset shape: `X = (8498, 1, 130, 79)`

#### 3. Preprocessing & Encoding
- **Label encoding:** `sklearn.LabelEncoder` → integer classes → `to_categorical` (6-class one-hot).
- **Train/Test Split:** 80% / 20% (`random_state=102`, shuffle=True).
  - Training set: **6,798 samples** | Test set: **1,700 samples**
- **Standardization:** Z-score normalization using training set mean & std (applied to both train and test).
- **Reshape:** `(N, 1, 130, 79)` → `(N, 130, 79)` for CNN input.

#### 4. Model Architecture (Keras Sequential CNN — 3 Conv Blocks)

| Block | Layers | Filters / Units |
|-------|--------|-----------------|
| Block 1 | Conv2D → MaxPool2D → Conv2D → MaxPool2D → Dropout(0.20) | 32, 32 |
| Block 2 | Conv2D → MaxPool2D → Conv2D → MaxPool2D → Dropout(0.25) | 64, 64 |
| Block 3 | Conv2D → MaxPool2D → Conv2D → MaxPool2D → Dropout(0.30) | 128, 128 |
| Head | Flatten → Dense(256, ReLU) → Dropout(0.50) → Dense(6, Softmax) | — |

- **Input shape:** (130, 79, 1)
- **Total parameters:** 320,998 (1.22 MB)
- **Optimizer:** Adam | **Loss:** Categorical Cross-Entropy | **Epochs:** 50 | **Batch size:** 32
- **Callback:** `EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)`

#### 5. Training
- Validation split of 20% from training data used during `model.fit`.
- Training accuracy started at ~62.8% (epoch 1), val accuracy at ~55.5%.

### Results

**Test Set Performance:**

| Metric | Value |
|--------|-------|
| Test Accuracy | **54.88%** |
| Test Loss | 1.6181 |

**Per-Emotion Classification Report (on 1,700 test samples):**

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| angry | 0.68 | 0.71 | **0.69** | 298 |
| disgust | 0.53 | 0.42 | 0.47 | 283 |
| fear | 0.53 | 0.43 | 0.47 | 270 |
| happy | 0.48 | 0.56 | 0.52 | 300 |
| neutral | 0.58 | 0.64 | **0.61** | 250 |
| sad | 0.50 | 0.54 | 0.52 | 299 |
| **Overall** | **0.55** | **0.55** | **0.55** | **1,700** |

**Sample Predictions (rows 140–144 of test set):**

| Index | Predicted | Actual |
|-------|-----------|--------|
| 140 | angry | happy ✗ |
| 141 | fear | fear ✓ |
| 142 | neutral | neutral ✓ |
| 143 | happy | happy ✓ |
| 144 | angry | angry ✓ |

**Confusion Matrix** generated as a Seaborn heatmap (`cmap='magma'`).

**Conclusion:** The model achieves **54.88% accuracy** on a 6-class speech emotion task — well above random chance (16.7%). `angry` and `neutral` are the best-recognized emotions (F1 = 0.69, 0.61). `disgust` and `fear` are the hardest to distinguish. Improvements can be explored through data augmentation (pitch shift, time stretch, noise injection), attention mechanisms, or pre-trained audio models (Wav2Vec2, HuBERT).

---

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Data & EDA | `pandas`, `numpy`, `matplotlib`, `seaborn` |
| Audio Processing | `librosa`, `librosa.display` |
| Machine Learning | `scikit-learn` (LinearRegression, LogisticRegression, LabelEncoder, train_test_split, metrics) |
| Deep Learning | `tensorflow` / `keras` (Sequential, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, EarlyStopping, to_categorical) |
| Environment | Google Colab (GPU runtime; datasets loaded from Google Drive) |

---

## Repository Structure

```
Emot_Boy/
├── 01_Linear_Regression_Project.ipynb    # Ecommerce yearly spend prediction
├── 02_Logistic_Regression_Project.ipynb  # Ad click binary classification
├── CNN_on_Animals.ipynb                  # 15-class wildlife image classifier
├── Librosa_CNN.ipynb                     # 6-class speech emotion recognition
└── README.md
```
