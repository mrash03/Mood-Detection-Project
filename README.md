# FER2013 Emotion Recognition Using Transfer Learning with CNN

## Objective
Classify facial emotions from grayscale images using a pre-trained CNN and transfer learning.

## Dataset
FER2013: 35,887 grayscale images (48x48 pixels) labeled with 7 emotions:
- 0: Angry
- 1: Disgust
- 2: Fear
- 3: Happy
- 4: Sad
- 5: Surprise
- 6: Neutral

## Method
- Used a pre-trained CNN (e.g., ResNet50/VGG16) as the base model.
- Applied Transfer Learning: froze base layers and fine-tuned top layers on FER2013.
- Data augmentation applied: rotation, shift, zoom, horizontal flip.
- Used Test-Time Augmentation (TTA) to improve accuracy.

## Results

### Without TTA
- Accuracy: 69.7%
- Macro F1-score: 0.6769

### With TTA
- Accuracy: 70.37%
- Macro F1-score: 0.68

### Classification Report (TTA)
| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.66      | 0.56   | 0.61     | 991     |
| 1     | 0.63      | 0.68   | 0.65     | 109     |
| 2     | 0.64      | 0.48   | 0.55     | 1024    |
| 3     | 0.89      | 0.88   | 0.89     | 1798    |
| 4     | 0.58      | 0.63   | 0.61     | 1216    |
| 5     | 0.75      | 0.86   | 0.80     | 800     |
| 6     | 0.62      | 0.72   | 0.66     | 1240    |

## Confusion Matrix
![Confusion Matrix]

## Training & Validation Graphs
![Accuracy & Loss Curves]

## How to Run
- Open the notebook `FER2013_Transfer_Learning.ipynb` in Kaggle or local Jupyter.
- Run all the cells. 
