# Proyecto sobre un Chatbot de Salud Mental - Backend

## Funcionalidades Principales

1. **Carga del dataset**: El chatbot comienza cargando un dataset que contiene ejemplos de conversaciones sobre salud mental. Este dataset se utiliza para entrenar el modelo de inteligencia artificial del chatbot.

2. **Tokenización**: El chatbot procesa el texto del dataset, dividiéndolo en "tokens" individuales (por ejemplo, palabras). Este es un paso crucial en el procesamiento del lenguaje natural.

3. **Creación del modelo**: El chatbot utiliza un modelo de red neuronal para entender y responder a las preguntas. Este modelo se crea y se configura durante esta etapa.

4. **Entrenamiento del modelo**: Una vez que el modelo ha sido creado, se entrena utilizando el dataset. Durante este proceso, el modelo aprende a asociar ciertos patrones de palabras con ciertas respuestas.

5. **Guardar el modelo**: Una vez que el chatbot ha sido entrenado, el modelo se guarda para su uso futuro. Esto significa que no necesitamos reentrenar el modelo cada vez que queramos usar el chatbot. En su lugar, podemos cargar el modelo guardado, lo que ahorra tiempo y recursos computacionales. Además, el modelo guardado puede ser compartido y reutilizado por otros, permitiendo la reproducibilidad y la colaboración.

6. **Validación del modelo (opcional)**: La validación del modelo es un paso opcional que utiliza datos de prueba para evaluar su rendimiento. Esto nos permite asegurarnos de que el modelo es capaz de hacer predicciones precisas sobre datos nuevos.

7. **Despliegue del chat**: Una vez que el modelo es cargado, se utiliza para interactuar con el usuario. Basado en un input del usuario, se genera una respuesta adecuada.

## Instalación

### Pre-requisitos
- Python 3.x
- pip

### Pasos
1. Clonar el repositorio:
   ```sh
   git clone <URL_DEL_REPOSITORIO>
   cd <NOMBRE_DEL_REPOSITORIO>
   ```

2. Instalar dependencias del backend:
   ```sh
   pip install -r requirements.txt
   ```

## Uso

### Scripts principales
1. Ejecutar el entrenamiento de la IA:
   ```sh
   python training.py
   ```

2. Probar el chatbot en consola:
   ```sh
   python chat.py
   ```

### Opcional
1. Validar el modelo con datos de prueba (opcional):
   ```sh
   python validate.py
   ```

### Despliegue como API
Si deseas usar el chatbot como una API, puedes ejecutar el siguiente script:
   ```sh
   python app.py
   ```