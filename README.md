
# Gender Classification on ESP32 using TinyML

This project demonstrates real-time gender classification using a neural network model deployed on an ESP32 microcontroller. Leveraging TinyML and TensorFlow Lite, the system performs on-device inference using demographic features like age, height, weight, etc., without relying on the cloud.

---

## 🔧 Features

- Edge inference on ESP32
- Gender classification (Male/Female)
- Neural network built with TensorFlow/Keras
- Model conversion to TFLite and deployment using EloquentTinyML

---

## 🧠 Dataset

- **Features**: Age, Height, Weight, Occupation, Education, Marital Status, Favorite Color
- **Target**: Gender (Male = 1, Female = 0)
- **Preprocessing**:
  - Normalization of numerical features
  - Label encoding of categorical features
  - Train-test-validation split: 70%-15%-15%

---

## 🧱 Model Architecture

```python
model = Sequential([
    Dense(128, activation='sigmoid', input_shape=(8,)),
    Dense(64, activation='relu'),
    Dense(32, activation='sigmoid'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Accuracy: ~92%

---

## 🖥️ Requirements

### Hardware
- ESP32 microcontroller

### Software
- Python 3.x
- TensorFlow, NumPy, Pandas, Scikit-learn
- Arduino IDE
- EloquentTinyML library

---

## 🚀 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/gender-classification-esp32.git
cd gender-classification-esp32
```

### 2. Preprocess and Train the Model (Python)
```bash
pip install tensorflow pandas numpy scikit-learn
python model_training.py
```

### 3. Convert Model to TensorFlow Lite
```python
# inside model_training.py
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
```

### 4. Convert `.tflite` to `.h` file
```bash
xxd -i model.tflite > model.h
```

Move `model.h` to the Arduino project folder.

---

## ⚙️ Deploy on ESP32

### 1. Install Arduino IDE and Required Libraries
- Install [EloquentTinyML](https://github.com/eloquentarduino/EloquentTinyML) via Library Manager.

### 2. Upload Code
- Open `gender_classifier.ino` in Arduino IDE.
- Include `model.h` in the sketch.
- Upload to your ESP32 board.

```cpp
#include "model.h"
#include <EloquentTinyML.h>

#define NUMBER_OF_INPUTS 8
#define NUMBER_OF_OUTPUTS 1
#define TENSOR_ARENA_SIZE 16*1024

Eloquent::TinyML::TfLite<NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE> ml;

void setup() {
  Serial.begin(115200);
  ml.begin(model);
}

void loop() {
  float input[8] = {0.25, 0.7, 0.55, 1, 2, 1, 0, 3}; // Example normalized values
  float prediction = ml.predict(input);
  Serial.print("Prediction: ");
  Serial.println(prediction > 0.5 ? "Male" : "Female");
  delay(2000);
}
```

---

## ✅ Output

- Inference on ESP32 through Serial Monitor:
```
Prediction: Female
Prediction: Male
```

---

## 📈 Results

- Accuracy: 92% on test dataset
- Real-time inference on ESP32 with minimal latency
- Efficient memory usage using Tensor Arena (16KB)

---

## 📚 Learnings & Future Work

- Learned about deploying machine learning models on low-resource devices.
- Future work may include:
  - Quantization-aware training
  - Optimizing inference speed
  - Extension to other classifications (e.g., age group, BMI category)

---

## 📄 License

BSD 3-Clause License

Copyright (c) 2025, PrathamK

## Output
![Output Screenshot](https://github.com/user-attachments/assets/305babc7-a0eb-4b8a-ba13-b9357ca9a5a3)

