# Skills

## Level 1: Metadata

* **Name:** Manas Malviya
* **Domain:** Computer Science & Data Science
* **Current Status:** 3rd Year Undergraduate (BMS College of Engineering)
* **CGPA:** 9.69
* **Focus Areas:** Machine Learning, Deep Learning, Data Analysis, Artificial Intelligence, Software Development
* **Primary Languages:** Python, C++, Java
* **Core Strengths:** Problem Solving, Data Structures & Algorithms, ML/DL Model Development

---

## Level 2: Instructions

### 🔧 Programming & Development

* Write clean, modular, and efficient code in Python, C++, and Java
* Apply object-oriented programming concepts in real-world applications
* Implement data structures and algorithms for optimized solutions

### 🗄️ Databases

* Work with relational databases:

  * MySQL
  * PostgreSQL
* Perform:

  * CRUD operations
  * Joins, indexing, and query optimization
* Work with NoSQL databases:

  * CouchDB
* Handle:

  * JSON-based document storage
  * REST-based database interactions

### 🤖 Machine Learning & Data Science

* Perform data preprocessing:

  * Data cleaning
  * Tokenization
  * Normalization
* Apply machine learning algorithms:

  * Linear Regression
  * Logistic Regression
  * Random Forest
  * Gradient Boosting
* Use feature engineering techniques:

  * TF-IDF (for NLP tasks)
  * Technical indicators (for time series)
* Evaluate models using:

  * Accuracy
  * MAE
  * R² Score

### 🧠 Deep Learning

* Understand neural network fundamentals:

  * Activation functions, backpropagation
* Build models using:

  * ANN
  * CNN 
* Use frameworks:

  * TensorFlow / Keras
* Handle overfitting:

  * Dropout, regularization

### 📊 Data Analysis & Visualization

* Analyze structured and unstructured datasets
* Create visualizations using:

  * Matplotlib
  * Plotly
  * Power BI

### ⚙️ Tools & Workflow

* VS Code, PyCharm, Jupyter Notebook
* Google Colab
* Flask, Streamlit

---

## Level 3: Resources & Code

### 🐍 Python (Data Analysis)

```python
import pandas as pd
df = pd.read_csv("data.csv")
print(df.head())
```

---

### 🗄️ SQL (PostgreSQL / MySQL)

```sql
SELECT name, salary
FROM employees
WHERE salary > 50000
ORDER BY salary DESC;
```

---

### 📦 CouchDB (NoSQL Example)

```json
{
  "_id": "user_001",
  "name": "Manas",
  "skills": ["Python", "ML", "SQL"],
  "experience": "student"
}
```

---

### 🔄 CouchDB (HTTP API Example)

```bash
curl -X GET http://127.0.0.1:5984/mydatabase/user_001
```

---

### 🤖 Machine Learning

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)
```

---

### 🧠 Deep Learning (Keras)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(1, activation='sigmoid')
])
```

---

### 📊 Visualization

```python
import matplotlib.pyplot as plt

plt.plot([1,2,3], [4,5,6])
plt.show()
```

---


---

## 🧠 Deep Learning Advanced (TensorFlow / Keras)

### 🔹 Basic Image Classification (MNIST)

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, validation_split=0.2)
```

---

### 🔹 Convolutional Neural Network (CNN)

```python
def generate_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), padding='same', input_shape=(32,32,3)),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2,2)),

        tf.keras.layers.Conv2D(64, (3,3), padding='same'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2,2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
```

---

### 🔹 Custom Dense Layer

```python
import tensorflow as tf

class CustomDense(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units))
        self.b = self.add_weight(shape=(self.units,))

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
```

---

### 🔹 Residual Block (ResNet Concept)

```python
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, 3, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters, 3, padding='same')

    def call(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        return tf.nn.relu(x + y)
```

---

### 🔹 Multi-Task Learning Model

```python
class MultiTaskModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.shared = tf.keras.layers.Dense(128, activation='relu')
        self.task1 = tf.keras.layers.Dense(10, activation='softmax')
        self.task2 = tf.keras.layers.Dense(5, activation='softmax')

    def call(self, x):
        x = self.shared(x)
        return self.task1(x), self.task2(x)
```

---

### 🔹 Custom GRU Cell (RNN)

```python
import tensorflow.experimental.numpy as tnp

class GRUCell:
    def __init__(self, n_units):
        self.n_units = n_units
```

*(Full implementation included in project reference)* 

---

---

## 🚀 Model Deployment (TensorFlow)

### 🔹 Overview

Deploy trained models to production using:

* TensorFlow SavedModel
* TensorFlow Lite (mobile/edge)
* Quantization techniques
* Serving infrastructure (REST / Docker)

---

### 🔹 SavedModel Export

```python
model.save('saved_model')
loaded_model = tf.keras.models.load_model('saved_model')
predictions = loaded_model.predict(test_data)
```

---

### 🔹 TensorFlow Lite Conversion

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

---

### 🔹 Quantization (Model Optimization)

```python
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

---

### 🔹 TFLite Inference

```python
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
```

---

### 🔹 TensorFlow Serving (Production)

```bash
docker run -p 8501:8501 \
  --mount type=bind,source=/model,target=/models/my_model \
  -e MODEL_NAME=my_model \
  tensorflow/serving
```

---

---

## 🛠️ TensorFlow Debugging & Optimization

### 🔹 Overview

Systematic debugging of deep learning models:

* Shape mismatches
* Memory (OOM) issues
* NaN/Inf losses
* Gradient problems
* GPU configuration errors

---

### 🔹 Shape Debugging

```python id="7yqk9f"
print(x.shape)
tf.debugging.assert_shapes([(x, ('batch', 'features'))])
```

---

### 🔹 Memory (OOM) Handling

```python id="a9r5kn"
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

---

### 🔹 NaN / Loss Issues

```python id="9g5dbe"
tf.debugging.check_numerics(tensor, "NaN detected")

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
```

---

### 🔹 Gradient Debugging

```python id="2t6k2g"
with tf.GradientTape() as tape:
    loss = loss_fn(y, model(x))

grads = tape.gradient(loss, model.trainable_variables)
```

---

### 🔹 TensorBoard Debugging

```python id="a7c2d1"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs')
model.fit(x_train, y_train, callbacks=[tensorboard_callback])
```

---

### 🔹 Data Pipeline Optimization

```python id="w6n1pr"
dataset = dataset.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
```

---

### 🔹 When to Use

* Debug training failures
* Optimize model performance
* Fix data pipeline bottlenecks
* Improve training stability

---

### ⚠️ Best Practices

* Validate tensor shapes early
* Monitor gradients and loss
* Use TensorBoard for debugging
* Normalize inputs
* Profile performance

---

### ❗ Common Issues

* Shape mismatch errors
* NaN loss (high learning rate)
* GPU not detected
* Data pipeline bottlenecks
* Vanishing/exploding gradients

---

*(Full debugging workflows and advanced diagnostics included in reference)* 

---


### 🔹 When to Use

* Deploy ML models to production
* Mobile / Edge AI applications
* Optimize model size & latency
* Build ML APIs

---

### ⚠️ Best Practices

* Validate converted models
* Use SavedModel format
* Apply quantization carefully
* Test on target device
* Version your models

---

### ❗ Common Pitfalls

* Not validating TFLite outputs
* Over-quantization → accuracy drop
* Ignoring device constraints
* No model versioning
* Wrong input/output shapes

---

*(Full deployment pipelines and advanced optimizations implemented in project references)* 

---


## ⚡ When to Use Deep Learning

* Image classification (CNN)
* Text processing (RNN / NLP)
* Time-series forecasting
* Multi-task learning problems

---

## ⚠️ Best Practices

* Normalize input data
* Use Dropout to avoid overfitting
* Monitor validation metrics
* Use appropriate loss functions
* Start simple → then increase complexity

---


---

## 📚 Resources

### 🔹 Official Documentation

* TensorFlow Keras Guide
* Sequential Model API
* Model Subclassing Guide
* RNN Guide
* Custom Training Loops

---

### 🔹 Learning Topics Covered

* Neural Networks (ANN, CNN, RNN)
* Custom Layers and Model Subclassing
* Multi-task Learning
* Residual Connections (ResNet)
* Feature Engineering for Deep Learning
* Model Optimization and Regularization

---

### 🔹 Advanced Concepts

* Transfer Learning
* Attention Mechanisms
* Time Series Forecasting
* Generative Models (Autoencoders, GANs)
* Reinforcement Learning Basics

---

### 🔹 Best Practices

* Normalize input data before training
* Use Dropout (0.2–0.5) to prevent overfitting
* Monitor validation metrics during training
* Choose correct loss functions:

  * `categorical_crossentropy` (one-hot labels)
  * `sparse_categorical_crossentropy` (integer labels)
* Use `model.summary()` to verify architecture
* Implement EarlyStopping for efficient training

---

### 🔹 Common Pitfalls

* Not normalizing input data
* Using incorrect loss functions
* Overfitting on small datasets
* Learning rate too high
* Not using validation data
* Dimension mismatch between layers

---

### 🔹 Useful Links

* https://www.tensorflow.org/
* https://keras.io/
* https://www.kaggle.com/
* https://scikit-learn.org/

---


