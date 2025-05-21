# CNN_Waste_Segregation

## Overview
This project implements a Convolutional Neural Network (CNN) to classify waste images into various categories, enabling efficient and automated waste segregation. The goal is to support better waste management by sorting waste into recyclable, non-recyclable, and other specific classes.

## Objective
Develop an effective waste segregation system using CNNs to accurately classify waste materials such as cardboard, glass, paper, plastic, food waste, metal, and others. This aims to improve recycling efficiency, reduce landfill waste, and promote sustainable waste management practices.

## Dataset
- Contains images categorized into **7 classes**:
  - Cardboard
  - Food Waste
  - Metal
  - Paper
  - Plastic
  - Glass
  - Other
- Each folder contains images related to its class but without finer subcategories (e.g., Food Waste includes various items like coffee grounds, teabags, fruit peels).
- Data preprocessing steps:
  - Image resizing
  - Normalization
  - One-hot encoding of labels
- Data split:
  - 80% training
  - 20% validation
  - Stratified splitting to maintain class balance
    
![image](https://github.com/user-attachments/assets/19a39b6b-b089-4fb0-a8e4-8ac90d37d30b)
    
![image](https://github.com/user-attachments/assets/0be2bb76-0907-4577-9f96-70faeb5628db)


## Model Architecture
- Multi-layer CNN with convolutional, pooling, dropout, and dense layers
- ReLU activation in hidden layers; softmax activation for output layer
- Compiled with categorical cross-entropy loss and Adam optimizer

## Training
- Trained on the training set, validated on the validation set
- Optionally uses early stopping and checkpointing to avoid overfitting
- Metrics tracked: accuracy and loss

## Evaluation
- Evaluated on a held-out test dataset
- Achieved approximately **44% test accuracy**
- Classification report and confusion matrix detail precision, recall, and F1-score per class
- Some classes show imbalance and misclassification, suggesting areas for improvement

## Environment & Libraries
- Python 3.x
- numpy: 1.26.4  
- pandas: 2.2.2  
- seaborn: 0.13.2  
- matplotlib: 3.8.4  
- PIL (Pillow): 10.3.0  
- tensorflow: 2.19.0  
- keras: 3.10.0  
- scikit-learn: 1.4.2  

## Future Work and Improvements
- Address class imbalance with data augmentation or class weighting.
- Experiment with transfer learning using pretrained CNN models.
- Hyperparameter tuning to improve accuracy and reduce loss.
- Deploy the model into a web or mobile application for real-time waste segregation.

## License
Licensed under the MIT License © 2025 Vaishali — Free to use, modify, and distribute with proper attribution.  
Provided "as is" without warranty for the CNN Waste Segregation project.

## Contact
For questions or collaboration, please contact:  
Vaishali Makwana  
Email: vaishali@vaishalimakwana.com

---

Thank you for checking out the CNN Waste Segregation project!

