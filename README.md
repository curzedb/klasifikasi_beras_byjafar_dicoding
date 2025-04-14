# KLASIFIKASI 5 JENIS BERAS MENGGUNAKAN ALGORITMA CNN
**Proyek Klasifikasi untuk membedakan gambar Beras Arborio, Basmati, Ipsala, Jasmine, & Karacadag Menggunakan Algoritma CNN**

📌 **Dibuat untuk**: Submission Proyek 2(Akhir) kelas Dicoding X IDCamp - Machine Learning Intermediate(Menengah)

-------------------------------------------------------------------


## 📝 Deskripsi Proyek
Proyek ini mengimplementasikan **Convolutional Neural Network (CNN)** untuk mengklasifikasikan gambar beras berdasarkan 5 kategori:
- Beras Arborio 
- Beras Basmati 
- Beras Ipsala
- Beras Jasmine
- Beras Karacadag

Dataset yang digunakan: <a href="https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset">rice.zip</a> (70.000+ Gambar).

Sitasi:
<br>
Koklu, M., Cinar, I., & Taspinar, Y. S. (2021). Classification of rice varieties with deep learning methods. Computers and Electronics in Agriculture, 187, 106285. https://doi.org/10.1016/j.compag.2021.106285

-----------------------------------------------------------------------
## 🛠️ Tools
- **Bahasa Pemrograman**: Python 3
- **Framework**: TensorFlow
- **Libraries**: Matplotlib, NumPy, os, pandas, seaborn, sklearn, pathlib, PIL, seaborn, tensorflow, zipfile
- **Platform**: Google Colab

-----------------------------------------------------------------------
## 📊 Struktur Dataset  
```
dataset/  
  ├── Arborio/
  ├── Basmati/
  ├── Ipsala/
  ├── Jasmine/
  └── Karacadag/
```

### Jenis Beras:
1. Arborio<br>
![image](https://github.com/user-attachments/assets/362e3c80-cb11-4a6e-8cc5-4446e49741d7)
2. Basmati<br>
![image](https://github.com/user-attachments/assets/d03e13b6-d6b1-4105-8db7-adb96cbae49e)
3. Ipsala<br>
![image](https://github.com/user-attachments/assets/657a77c0-ee52-4563-914a-a6e41c2eb0cf)
4. Jasmine<br>
![image](https://github.com/user-attachments/assets/42fe4f67-4a10-415d-80b7-71e7de2831e9)
5. Karacadag<br>
![image](https://github.com/user-attachments/assets/24e5a2c9-a57f-48c9-b3c3-f15f47e76462)

### Pembagian Data:
| Jenis Data  | Persentase Data |
|-------------|-----------------|
| Data Latih  | 80%             |
| Data Uji    | 20%             |

---------------------------------------------------------------------
## 🧠 Arsitektur Model CNN  
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, activation='softmax')
    ])

model.compile(loss = 'categorical_crossentropy',
              optimizer=tf.optimizers.Adam(),
              metrics=['accuracy'])
model.summary()
```

**Hyperparameter**:  
- Optimizer: `Adam`  
- Loss: `Categorycal_Crossentropy`
- Model: `Sequential`  
- Epochs: `16`  
- Batch Size: `32`

-----------------------------------------------------------------------
## 📈 Hasil Evaluasi  
| Metric      | Validation |
|-------------|------------|
| Accuracy    | 97.16%     |
| Loss        | 0.084      |

### Model Akurasi:<br> 
![image](https://github.com/user-attachments/assets/25405419-1973-4328-8f34-65289778593d)

### Model Loss:<br>
![image](https://github.com/user-attachments/assets/360ac725-f702-4f02-abaf-c8be879e9f79)


-----------------------------------------------------------------------
## ☑️Implementasi
```python
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="beras_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define the image path for inference
image_path = '/content/dataset_beras/Karacadag/Karacadag (10).jpg'

# Preprocess the image
img = load_img(image_path, target_size=(150, 150))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x / 255.0

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], x)

# Run inference
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

# Get the predicted class
predicted_class = np.argmax(output_data)

# Define class labels
class_labels = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

# Print the prediction
print(f"Predicted class: {class_labels[predicted_class]}")

# Display the image
plt.imshow(img)
plt.title(f"Prediction: {class_labels[predicted_class]}")
plt.axis('off')
plt.show()
```
Hasil: <br>
![image](https://github.com/user-attachments/assets/43a4fb1d-b845-46a0-893f-9b5a6cb79255)



