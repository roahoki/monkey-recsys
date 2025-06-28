# Sistema de Recomendación de Películas con Gemma 2B

Este documento analiza el cuaderno `main_Gemma.ipynb`, que implementa un sistema de recomendación de películas utilizando el modelo de lenguaje Gemma 2B.

## Resumen General

El cuaderno implementa un sistema que genera recomendaciones de películas basándose en conversaciones de usuarios. Se evalúan tres estrategias principales:

* **Zero-shot**: Recomendaciones sin entrenamiento específico para la tarea
* **Few-shot**: Recomendaciones con ejemplos de contexto
* **Fine-tuning**: Recomendaciones con modelo especializado mediante entrenamiento

## Análisis Paso a Paso

### 1. Configuración Inicial y Carga de Datos

**Celdas 1-5: Configuración para Google Colab**
* Se monta Google Drive para acceder al dataset
* Se instalan las librerías necesarias: 
  * unsloth
  * xformers
  * trl
  * peft
  * accelerate
  * bitsandbytes
  * rapidfuzz
  * wandb
* Se copian archivos necesarios como `Tools.py`
* Se extrae el dataset comprimido (LLM_Redial.zip)

**Celdas 6-7: Configuración del modelo Gemma**
* Se autentica con Hugging Face para acceder al modelo
* Se inicializa el tokenizador para el modelo "google/gemma-2b-it"

**Celda 8: Carga de datos del dataset**
* Se cargan los datos del dataset para la categoría "Movie"
* Se leen los archivos: 
  * final_data.jsonl
  * Conversation.txt
  * user_ids.json
  * item_map.json
* Estos contienen las conversaciones y mapeos de IDs a nombres de películas
### 2. Funciones Utilitarias

**Celdas 9-10: Limpieza del entorno**
* Define funciones para limpiar variables pesadas y liberar memoria
* Útil para sesiones largas en entornos con recursos limitados
* Incluye `cleanup_for_dill_serialization()` y `cleanup_after_lora_generation()`

**Celda 11-12: Procesamiento de respuestas**
* Implementa funciones para formatear las respuestas del modelo:
  * `format_ans()`: Extrae nombres de películas desde las respuestas generadas
  * `extraer_listas_recomendadas()`: Procesa las salidas del modelo para extraer listas de películas

**Celda 13-14: Persistencia de datos**
* Funciones para guardar y cargar respuestas del modelo en formato JSON:
  * `guardar_datos_json()`: Guarda listas de resultados en archivos JSON
  * `cargar_datos_json()`: Carga resultados previamente guardados
* Permite retomar el trabajo sin tener que regenerar todas las respuestas
### 3. Preparación de Datos para Entrenamiento y Evaluación

**Celdas 15-16: Carga de conversaciones**
* Implementa la función `load_conversations()` para cargar y estructurar todas las conversaciones
* Cada conversación se organiza por ID para facilitar su procesamiento posterior
* Permite acceder a todas las conversaciones almacenadas en el dataset

**Celdas 17-18: División de datos**
* Se separan los diálogos en tres conjuntos:
  * Entrenamiento (80%)
  * Prueba (10%) 
  * Validación (10%)
* Se asegura que todas las conversaciones de un mismo usuario queden en un único conjunto
* Esta estrategia evita la filtración de información entre conjuntos, mejorando la validez de las evaluaciones

**Celdas 19-20: Selección de diálogos para prueba**
* Implementa la función `extraer_dialogos()` para seleccionar aleatoriamente conversaciones
* Se seleccionan `num_test_items` (100) conversaciones para evaluar los modelos
* Se configura una opción interactiva para:
  * Generar nuevas respuestas (opción 1)
  * Cargar respuestas anteriores (opción 2)
* Incluye sistema para guardar/cargar resultados con identificación por "seed"
### 4. Generación de Recomendaciones

#### **Celdas 21-22: Zero-Shot con interacciones históricas**

* **Método**: Implementa el enfoque zero-shot usando el modelo Gemma sin entrenamiento específico
* **Proceso**:
  * Para cada conversación de prueba:
    * Crea un prompt combinando la conversación y las interacciones del usuario
    * Genera 20 listas diferentes de 10 películas cada una
    * Utiliza sampling con temperatura 0.7 para diversificar las recomendaciones
  * Se registra el tiempo de generación para monitorear el rendimiento
  * Las respuestas se guardan en `outputs_z_s` y se procesan con `extraer_listas_recomendadas()`

#### **Celdas 23-24: Few-Shot con interacciones históricas**

* **Selección de ejemplos**:
  * Se eligen aleatoriamente 5 usuarios del conjunto de entrenamiento
  * Se extraen sus conversaciones, interacciones y preferencias
  * Estos datos crean ejemplos de referencia para el modelo

* **Generación**:
  * Se construye un prompt enriquecido con ejemplos + conversación de prueba
  * Se generan múltiples listas de recomendaciones (20 por conversación)
  * Se utiliza la misma configuración de generación que en zero-shot

#### **Celdas 25-30: Fine-Tuning**

* **Preparación de datos**:
  * Se calculan las 20 películas más populares entre los usuarios de entrenamiento
  * Se crea una función `preparar_datos_fine_tuning()` que construye ejemplos de calidad:
    * Incluye la película recomendada en el ground truth
    * Agrega películas que le gustan al usuario (no mencionadas en la conversación)
    * Incorpora películas de las interacciones históricas
    * Complementa con películas populares hasta completar 10 recomendaciones

* **Implementación del fine-tuning**:
  * Utiliza LoRA (Low-Rank Adaptation) para eficiencia en el entrenamiento
  * Configura parámetros optimizados:
    * Batch size de 4 con acumulación de gradientes (8 pasos)
    * Learning rate de 2e-4 con programación coseno
    * 500 pasos de entrenamiento
    * Precisión BF16 para mejor rendimiento

#### **Celda 31: Generación con modelo fine-tuned**

* Carga el modelo fine-tuned desde la ruta específica
* Genera recomendaciones usando este modelo personalizado
* Mantiene la misma estructura de output que en los otros enfoques para comparación justa
### 5. Evaluación de Resultados

#### **Celdas 32-34: Funciones de evaluación**

* **Métricas implementadas**:
  * **Recall@k**: Mide si la película recomendada correctamente aparece entre las k primeras recomendaciones
  * **NDCG@k**: Evalúa tanto la presencia como el ranking de las recomendaciones relevantes

* **Preprocesamiento de títulos**:
  * Función `normalizar_titulo()`: Estandariza los títulos para comparación robusta
  * Función `comparar_titulos()`: Usa fuzzy matching para tolerancia a variaciones
  * Umbral de similitud de 80% para considerar coincidencias

* **Funciones auxiliares**:
  * `clean_lists()`: Procesa y estandariza las listas generadas por los modelos
  * `evaluate_recommendations()`: Calcula métricas sobre conjuntos completos
  * `evaluate_model()`: Función principal que reporta resultados formateados

#### **Celdas 35-36: Evaluación comparativa**

* **Procesamiento de resultados**:
  * Limpieza de todas las listas de recomendaciones generadas
  * Verificación de formato y conteo de elementos

* **Evaluación sistemática**:
  * Evalúa cada enfoque (Zero-Shot, Few-Shot y Fine-Tuning) 
  * Calcula métricas para dos escenarios:
    * k=5 (primeras 5 recomendaciones)
    * k=10 (todas las recomendaciones)
  * Presenta resultados formateados para comparación directa

#### **Celdas 37-41: Análisis de incertidumbre**

* **Técnicas implementadas**:
  * Input Clarification Ensembling: Analiza variaciones en respuestas ante paráfrasis
  * Análisis de logits: Examina la distribución de probabilidades en la capa de salida

* **Métricas de incertidumbre**:
  * **Incertidumbre total**: Entropía general de todas las respuestas
  * **Incertidumbre aleatoria**: Asociada a la ambigüedad inherente del problema
  * **Incertidumbre epistémica**: Relacionada con el conocimiento del modelo
  * **Consistencia entre paráfrasis**: Mide estabilidad de respuestas

* **Interpretación automática**:
  * Analiza dominancia entre incertidumbre epistémica/aleatoria
  * Evalúa nivel de consistencia y confianza del modelo

#### **Celda 42: Guardado del estado de sesión**

* Utiliza la biblioteca `dill` para guardar el estado completo
* Almacena en ruta `/content/gdrive/MyDrive/Proyecto LLMonkeys/sessions/Gemma/`
* Permite recuperar el trabajo en sesiones posteriores
## Relevancia del Proyecto

Este proyecto destaca por varias contribuciones significativas:

### 1. **Integración de Sistemas de Recomendación y LLMs**
* **Innovación**: Explora la aplicación de modelos de lenguaje grandes a problemas de recomendación
* **Contraste con métodos tradicionales**: Ofrece alternativa a sistemas basados en filtrado colaborativo o contenido
* **Aprovechamiento de contexto**: Utiliza la riqueza semántica de las conversaciones para mejorar recomendaciones

### 2. **Evaluación Sistemática de Estrategias**
* **Comparación metodológica**: Analiza tres estrategias (zero-shot, few-shot y fine-tuning) bajo las mismas condiciones
* **Métricas estandarizadas**: Emplea Recall@k y NDCG@k para evaluación cuantitativa
* **Protocolo replicable**: Establece un marco para futuras investigaciones en el área

### 3. **Recomendación en Contexto Conversacional**
* **Naturalidad**: Aborda las recomendaciones como parte natural de una conversación
* **Contextualización**: Aprovecha diálogos previos para entender preferencias implícitas
* **Experiencia de usuario**: Acerca los sistemas de recomendación a interacciones más humanas

### 4. **Optimización de Recursos mediante Fine-tuning Eficiente**
* **Técnicas avanzadas**: Implementa LoRA (Low-Rank Adaptation) para especialización eficiente
* **Democratización**: Permite adaptar modelos grandes con recursos computacionales limitados
* **Parámetros óptimos**: Documenta configuraciones efectivas para problemas similares

### 5. **Análisis de Incertidumbre en Recomendaciones**
* **Transparencia**: Proporciona métricas de confianza junto con las recomendaciones
* **Descomposición de incertidumbre**: Distingue entre limitaciones inherentes y del modelo
* **Mejora iterativa**: Establece bases para calibrar y refinar recomendaciones según la confianza