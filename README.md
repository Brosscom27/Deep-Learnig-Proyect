# Proyecto DIME-MEX 2025: Clasificación Multimodal de Discursos de Odio

Este proyecto forma parte del curso **DIME-MEX 2025** ([enlace oficial](https://sites.google.com/view/dimemex-2025/important-dates?authuser=0)) y tiene como objetivo desarrollar un modelo de clasificación de discursos de odio en memes, utilizando información tanto visual como textual.

## 📌 Descripción del Proyecto

El proyecto aborda el problema de detección automática de contenido ofensivo en memes, considerando la naturaleza multimodal del problema: imágenes con texto superpuesto que, en conjunto, pueden transmitir discursos de odio.

Nuestra primera solución combina técnicas de procesamiento de lenguaje natural y visión por computadora mediante el uso de:

- **BERT**: modelo de lenguaje preentrenado para extraer embeddings del texto.
- **CNN (Convolutional Neural Network)**: para procesar las imágenes y extraer sus características visuales.
- **Concatenación Multimodal**: se unen los embeddings de imagen y texto para ser clasificados por una capa densa final.
- **CLIP (Contrastive Language–Image Pre-training)**: modelo para tareas multimodales que aprende conceptos visuales de manera eficaz a partir de la supervisión del lenguaje natural.

## 🧠 Arquitectura

```text
[Imagen] -> CNN -> Embedding Visual
[Texto]  -> BERT -> Embedding Textual
      ↓             ↓
      ---- Concatenación ----
                 ↓
           Capa(s) Densas
                 ↓
         Predicción (Inofensivo / Inapropiado / Discurso de Odio)
