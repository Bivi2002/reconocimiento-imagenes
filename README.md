# Reconocedor de números escritos a mano

Modelo de IA en Python (TensorFlow/Keras) que reconoce dígitos del 0 al 9 a partir de una imagen, con una página web para probarlo.

## Archivos

- `imagenes.ipynb`: notebook de entrenamiento (Google Colab)
- `modelo.h5`: modelo entrenado
- `app.py`: servidor Flask con la ruta `/predecir`
- `templates/index.html`: página web para subir la imagen
- `requirements.txt`: dependencias

## Cómo ejecutarlo

1. Instalar las dependencias: `pip install -r requirements.txt`
2. Iniciar el servidor: `python app.py`
3. Abrir http://127.0.0.1:5000, sube una imagen de un número y pulsa **Analizar Imagen**.

## Preprocesamiento de la imagen

1. Convertir a escala de grises
2. Invertir colores
3. Redimensionar a 28x28 píxeles
4. Normalizar dividiendo entre 255

## Ejemplo de resultado

Número predicho: 2, con una confianza de 59.83%.
