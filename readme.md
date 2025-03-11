# ANN Model for Customer Churn Prediction

## 📌 Project Overview

This project implements an **Artificial Neural Network (ANN)** model from scratch to predict customer churn using the **Churn Modelling** dataset. The model is built using **TensorFlow/Keras** and includes feature preprocessing techniques such as **label encoding, one-hot encoding, and feature scaling**. It also integrates **TensorBoard** for visualization and early stopping to prevent overfitting.

## 📂 Dataset

The dataset used in this project is **Churn_Modelling.csv**, which contains customer information, account details, and churn labels.

## 🔧 Preprocessing Steps

1. **Dropping irrelevant columns**: Removed `RowNumber`, `CustomerId`, and `Surname`.
2. **Label Encoding**: Converted the `Gender` column to numeric form.
3. **One-Hot Encoding**: Transformed the `Geography` column.
4. **Feature Scaling**: Standardized numerical features using `StandardScaler`.
5. **Splitting Data**: Divided into training and test sets using `train_test_split`.
6. **Saving Encoders & Scaler**: Serialized using `pickle` for future inference.

## 🏗️ ANN Model Implementation

- The ANN is built using the **Sequential API** in Keras with the following layers:
  - **Input Layer**: Takes the scaled features as input.
  - **Hidden Layers**:
    - 64 neurons with ReLU activation
    - 32 neurons with ReLU activation
  - **Output Layer**: 1 neuron with Sigmoid activation (for binary classification).

### **Model Compilation & Training**

- Optimizer: `Adam` (learning rate = 0.01)
- Loss Function: `binary_crossentropy`
- Metrics: `accuracy`
- **Callbacks Used**:
  - `EarlyStopping`: Monitors validation loss and restores the best model.
  - `TensorBoard`: Logs training history for visualization.

## 🔥 Training the Model

```python
history = model.fit(
    X_train, y_train, validation_data=(X_test, y_test), epochs=100,
    callbacks=[tensorflow_callback, early_stopping_callback]
)
```

## 💾 Model & Preprocessing Save

- **Encoders & Scaler** saved as `.pkl` files.
- **Trained Model** saved as `model.h5`.

## 📊 TensorBoard Visualization

To visualize training logs:

```bash
%load_ext tensorboard
%tensorboard --logdir logs/fit
```

## 🚀 Key Learnings

✅ Implemented an **ANN model from scratch**.
✅ Preprocessed data with **Label Encoding, One-Hot Encoding, and Feature Scaling**.
✅ Used **Early Stopping** to prevent overfitting.
✅ Integrated **TensorBoard** for training insights.
✅ Saved and loaded the trained model for future inference.

## 📌 Next Steps

🔹 Experiment with **different hyperparameters** for better accuracy.
🔹 Try **other architectures like CNNs, LSTMs, or Transformers**.
🔹 Deploy the model using **Flask or FastAPI**.

---

💡 _This project is part of my Generative AI learning journey! Feel free to connect and discuss AI/ML ideas._ 🚀
