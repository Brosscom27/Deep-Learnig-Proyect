import os
import re
import json
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras import Input, Model
from tensorflow.keras.preprocessing import image
from transformers import AutoTokenizer, AutoModel
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications import ResNet50
from sklearn.metrics import accuracy_score, f1_score
from sentence_transformers import SentenceTransformer
from tensorflow.keras.applications.resnet50 import preprocess_input

ruta = 'C:/Users/52246/Documents/6to/Aprendizaje_Profundo/public_data_1/public_data_1/train_labels_tasks_1_3.csv'
ruta_images= 'C:/Users/52246/Documents/6to/Aprendizaje_Profundo/train/train'
ruta_images_validation = 'C:/Users/52246/Documents/6to/Aprendizaje_Profundo/validation/validation'

with open('C:/Users/52246/Documents/6to/Aprendizaje_Profundo/public_data_1/public_data_1/train_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

orden_imagenes = [item["MEME-ID"] for item in data]
print(orden_imagenes)

images = [os.path.join(ruta_images, nombre) for nombre in orden_imagenes]
images = [os.path.basename(path) for path in images]
print(images)
images_procesing = []
len(images)

with open('C:/Users/52246/Documents/6to/Aprendizaje_Profundo/public_data_1/public_data_1/validation_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

orden_imagenes_val = [item["MEME-ID"] for item in data]
print(orden_imagenes_val)

images_val = [os.path.join(ruta_images_validation, nombre) for nombre in orden_imagenes_val]
images_val = [os.path.basename(path) for path in images_val]
print(images_val)
images_procesing_val = []
len(images_val)

for img_file in images:
  image_temporal  = os.path.join(ruta_images, img_file)
  img = image.load_img(image_temporal, target_size = (128, 128))
  img = image.img_to_array(img) / 255.0
  images_procesing.append(img)

img_sample = images_procesing[0]

plt.imshow(img_sample)
plt.axis('off')
plt.show()

images_procesing = np.array(images_procesing)
labels = pd.read_csv(ruta, header = None).values

for img_file in images_val:
  image_temporal_val = os.path.join(ruta_images_validation, img_file)
  img_val = image.load_img(image_temporal_val, target_size = (128, 128))
  img_val = image.img_to_array(img_val) / 255.0
  images_procesing_val.append(img_val)

img_sample_val = images_procesing_val[0]

plt.imshow(img_sample_val)
plt.axis('off')
plt.show()

images_procesing_val = np.array(images_procesing_val)

base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

#Modelo TRAIN
def extraer_caracteristicas_resnet(ruta_img):
    img = Image.open(ruta_img).convert('RGB')
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)  

    features = base_model.predict(x)
    return features.flatten()  

embeddings_resnet = []

for nombre in tqdm(orden_imagenes):
    ruta = os.path.join(ruta_images, nombre)
    emb = extraer_caracteristicas_resnet(ruta)
    embeddings_resnet.append(emb)

X_imagenes = np.array(embeddings_resnet)

#Modelo De VALIDACIÓN 
def extraer_caracteristicas_resnet_val(ruta_img_val):
    img_vali = Image.open(ruta_img_val).convert('RGB')
    img_vali = img_vali.resize((224, 224))
    x = image.img_to_array(img_vali)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x) 

    features = base_model.predict(x)
    return features.flatten()  

embeddings_resnet_val = []

for nombre in tqdm(orden_imagenes_val):
    ruta_val = os.path.join(ruta_images_validation, nombre)
    emb_val= extraer_caracteristicas_resnet_val(ruta_val)
    embeddings_resnet_val.append(emb_val)

# Convertir a array final
X_imagenes_val = np.array(embeddings_resnet_val)
print(X_imagenes_val.shape)

#ROBERTA

# Modelo RoBERTa 
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
roberta = AutoModel.from_pretrained(model_name)
roberta.eval()

def obtener_embedding_roberta(texto):
    inputs = tokenizer(texto, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = roberta(**inputs)
        last_hidden_state = outputs.last_hidden_state  
        cls_embedding = last_hidden_state[:, 0, :]   
    return cls_embedding.squeeze().numpy()  

# Ruta al JSON
ruta_json = "C:/Users/52246/Documents/6to/Aprendizaje_Profundo/public_data_1/public_data_1/train_data.json"


with open(ruta_json, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Listas
nombres_imagenes = []
embeddings_desc = []

# Iterar sobre el JSON
for meme in tqdm(data):
    desc = meme.get("description", "")
    nombre = meme.get("MEME-ID", "")

    if desc and nombre:
        emb = obtener_embedding_roberta(desc)
        embeddings_desc.append(emb)
        nombres_imagenes.append(nombre)

# Ruta al JSON
ruta_json_val = 'C:/Users/52246/Documents/6to/Aprendizaje_Profundo/public_data_1/public_data_1/validation_data.json'


with open(ruta_json_val, 'r', encoding='utf-8') as f:
    data_val = json.load(f)

# Listas
nombres_imagenes_val = []
embeddings_desc_val = []

# Iterar sobre el JSON
for meme in tqdm(data_val):
    desc_val = meme.get("description", "")
    nombres_val = meme.get("MEME-ID", "")

    if desc and nombre:
        emb_val = obtener_embedding_roberta(desc_val)
        embeddings_desc_val.append(emb_val)
        nombres_imagenes_val.append(nombres_val)

combined_embeddings_robres = np.concatenate([X_imagenes, embeddings_desc], axis=1)

print(f"Shape de combined_embeddings: {combined_embeddings_robres.shape}")

combined_embeddings_robres_val = np.concatenate([X_imagenes_val, embeddings_desc_val], axis=1)

print(f"Shape de combined_embeddings: {combined_embeddings_robres_val.shape}")


# Entrada del vector concatenado
entrada_concatenada = Input(shape=(2816,), name='input_concatenado')

# Reducción de dimensionalidad
vector_reducido = Dense(1024, activation='relu', name='vector_reducido')(entrada_concatenada)

# Modelo de reducción
modelo_reduccion = Model(inputs=entrada_concatenada, outputs=vector_reducido)

nuevo_vector = modelo_reduccion.predict(combined_embeddings_robres)
print(len(nuevo_vector[0]))


X = nuevo_vector
y = np.argmax(labels, axis=1) 

kf = KFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []
f1_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n Fold {fold + 1}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # One-hot para entrenamiento
    num_clases = len(np.unique(y))
    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=num_clases)

    # Crear modelo fresco en cada fold
    input_dim = X.shape[1]
    entrada = tf.keras.Input(shape=(input_dim,))
    x = Dense(256, activation='relu')(entrada)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    salida = Dense(num_clases, activation='softmax')(x)

    modelo = tf.keras.Model(inputs=entrada, outputs=salida)
    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    modelo.fit(X_train, y_train_onehot, epochs=80, batch_size=128, verbose=0)

    # Evaluación
    y_pred_probs = modelo.predict(X_val)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')

    print(f"Accuracy: {acc:.4f}, F1 score: {f1:.4f}")

    accuracies.append(acc)
    f1_scores.append(f1)

    """cm = confusion_matrix(y_val, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión - Fold {fold + 1}')
    plt.xlabel('Predicho')
    plt.ylabel('Verdadero')
    plt.show()"""

# Resultados promedio
print("\n Resultados Promedio:")
print(f"Accuracy promedio: {np.mean(accuracies):.4f}")
print(f"F1-score promedio: {np.mean(f1_scores):.4f}")

