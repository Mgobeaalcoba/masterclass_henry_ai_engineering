# 🎯 Prompt Engineering: Generación de Scripts RAG

Este documento contiene ejemplos de **prompt engineering** para generar los tres scripts principales del proyecto usando un solo prompt cada uno.

---

## 📋 Índice

1. [Script sin Contexto](#1-script-sin-contexto-model_without_contextpy)
2. [Script con RAG](#2-script-con-rag-mainpy)
3. [Script Híbrido](#3-script-híbrido-main_hybridpy)

---

## 1. Script sin Contexto (`model_without_context.py`)

### 🎯 Objetivo
Crear un script simple que demuestre cómo un LLM **NO puede responder** sobre información privada porque no tiene acceso a documentación interna.

### 📝 Prompt Completo

```
Crea un script Python llamado `model_without_context.py` que demuestre la falta de contexto en LLMs.

OBJETIVO DEL SISTEMA:
El script debe demostrar que los modelos de lenguaje grandes (LLMs) NO pueden responder sobre información privada de empresas porque no tienen acceso a documentación interna. El sistema debe implementar un chat interactivo simple con OpenAI GPT que muestre claramente esta limitación educativa.

DEPENDENCIAS REQUERIDAS:
El proyecto utiliza Poetry como gestor de dependencias. Las dependencias necesarias son:
- Python 3.13 o superior
- openai 2.8.0 o superior
- python-dotenv 1.2.1 o superior

REQUISITOS FUNCIONALES:

1. DETECCIÓN AUTOMÁTICA DE MODELO:
   - El sistema debe detectar automáticamente qué modelo de OpenAI está disponible en la cuenta del usuario
   - Debe probar modelos en orden de preferencia: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo
   - Debe manejar errores cuando un modelo no está disponible y continuar con el siguiente
   - Debe informar al usuario qué modelo se está utilizando

2. CHAT INTERACTIVO:
   - El sistema debe proporcionar una interfaz de chat interactiva por terminal
   - Debe permitir al usuario hacer preguntas de forma continua hasta que decida salir
   - Debe reconocer comandos de salida: 'salir', 'exit', 'quit' (case-insensitive)
   - Debe mostrar mensajes informativos durante el proceso de consulta

3. CONFIGURACIÓN ANTI-ALUCINACIÓN:
   - El sistema debe configurar el modelo con temperatura muy baja (0.1) para minimizar creatividad y alucinaciones
   - Debe usar top_p de 0.9 para nucleus sampling más restrictivo
   - Debe limitar las respuestas a máximo 500 tokens
   - Debe incluir un prompt del sistema que instruya explícitamente al modelo a decir "No sé sobre..." cuando no tiene información, en lugar de inventar o suponer

4. GESTIÓN DE VARIABLES DE ENTORNO:
   - El sistema debe cargar la API Key de OpenAI desde un archivo .env
   - Debe validar que la API Key existe antes de iniciar el chat
   - Debe mostrar mensajes de error claros si falta la configuración

5. MANEJO DE ERRORES:
   - El sistema debe manejar errores de conexión, autenticación y otros errores de API
   - Debe mostrar mensajes de error amigables al usuario sin exponer detalles técnicos internos
   - Debe permitir continuar el chat después de un error

REQUISITOS TÉCNICOS:

1. ESTRUCTURA DEL ARCHIVO:
   - Debe incluir shebang para Python
   - Debe tener docstring principal que explique el propósito educativo del script
   - Debe organizarse en funciones claramente definidas

2. IMPORTS REQUERIDOS:
   - Módulo os del sistema estándar de Python
   - Clase OpenAI del paquete openai
   - Función load_dotenv del paquete dotenv

3. CONFIGURACIÓN DEL MODELO:
   - Temperature: 0.1 (muy baja para reducir alucinaciones)
   - max_tokens: 500
   - top_p: 0.9 (nucleus sampling restrictivo)
   - System prompt: Debe instruir al modelo a ser honesto y decir "No sé sobre..." cuando no tiene información

4. MENSAJES DE USUARIO:
   - Todos los mensajes deben incluir emojis apropiados para mejor UX
   - Debe mostrar claramente el nombre del modelo en uso
   - Debe incluir separadores visuales entre conversaciones
   - Mensajes deben estar en español

5. DOCUMENTACIÓN:
   - Todas las funciones deben tener docstrings en español
   - Comentarios en español cuando sean necesarios
   - Código debe seguir convenciones PEP 8

RESULTADO ESPERADO:
El script debe demostrar claramente que el modelo NO conoce datos privados de empresas, permitiendo al usuario hacer preguntas interactivamente y observando cómo el modelo responde honestamente cuando no tiene información, sin alucinar o inventar detalles.
```

### ✅ Características Clave del Prompt

- **Especifica el objetivo educativo**: "demostrar la falta de contexto"
- **Lista requisitos funcionales**: qué debe hacer el script
- **Lista requisitos técnicos**: librerías y configuraciones específicas
- **Define estructura**: funciones esperadas y organización
- **Incluye detalles específicos**: valores de temperatura, system prompt exacto

---

## 2. Script con RAG (`main.py`)

### 🎯 Objetivo
Crear un script que implemente RAG completo usando LangChain + ChromaDB para dar contexto al modelo sobre documentación privada.

### 📝 Prompt Completo

```
Crea un script Python llamado `main.py` que implemente un sistema RAG (Retrieval Augmented Generation) completo usando LangChain y ChromaDB.

OBJETIVO DEL SISTEMA:
El script debe demostrar cómo darle contexto a GPT sobre documentación privada usando LangChain + ChromaDB. El sistema debe permitir que el modelo responda sobre datos internos que no están en su entrenamiento mediante la implementación de un sistema RAG completo que carga documentación técnica, la procesa, y permite hacer preguntas interactivas que se responden usando el contexto de la documentación.

DEPENDENCIAS REQUERIDAS:
El proyecto utiliza Poetry como gestor de dependencias. Las dependencias necesarias son:
- Python 3.13 o superior
- langchain 1.0.7 o superior
- langchain-openai 1.0.3 o superior
- langchain-community 0.4.1 o superior
- langchain-huggingface 1.0.0 o superior (CRÍTICO: usar esta versión para evitar deprecaciones de HuggingFaceEmbeddings)
- chromadb 1.3.4 o superior
- python-dotenv 1.2.1 o superior
- openai 2.8.0 o superior
- sentence-transformers 5.1.2 o superior

REQUISITOS FUNCIONALES:

1. CARGA Y PROCESAMIENTO DE DOCUMENTACIÓN:
   - El sistema debe cargar un documento markdown desde el archivo "documentacion_tecnica.md"
   - Debe dividir el documento en fragmentos (chunks) de tamaño 500 caracteres con solapamiento de 50 caracteres
   - Debe crear embeddings vectoriales usando el modelo HuggingFace "sentence-transformers/all-MiniLM-L6-v2"
   - Debe almacenar los vectores en una base de datos vectorial ChromaDB persistente en el directorio "./chroma_db"
   - Debe mostrar mensajes informativos durante cada paso del proceso de configuración

2. SISTEMA DE RECUPERACIÓN (RETRIEVER):
   - El sistema debe configurar un retriever que busque los 3 fragmentos más relevantes para cada pregunta
   - Debe usar búsqueda por similitud semántica basada en embeddings
   - Debe informar al usuario cuántos fragmentos se consultaron para cada respuesta

3. CADENA RAG CON LANGCHAIN EXPRESSION LANGUAGE (LCEL):
   - El sistema debe implementar una cadena RAG usando LCEL (LangChain Expression Language)
   - La cadena debe combinar: recuperación de contexto → formateo de documentos → prompt template → modelo LLM → parser de salida
   - Debe usar el operador pipe de Python para componer la cadena
   - CRÍTICO: Debe usar la API moderna de LangChain (v1.0+), NO métodos deprecados

4. DETECCIÓN AUTOMÁTICA DE MODELO:
   - El sistema debe detectar automáticamente qué modelo de OpenAI está disponible
   - Debe probar modelos en orden de preferencia: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo
   - Debe manejar errores cuando un modelo no está disponible

5. CONFIGURACIÓN ANTI-ALUCINACIÓN:
   - El sistema debe configurar el modelo con temperatura 0.1 para minimizar alucinaciones
   - Debe limitar las respuestas a máximo 500 tokens
   - Debe incluir un prompt template estricto que instruya al modelo a usar SOLO el contexto proporcionado
   - El prompt debe indicar explícitamente que si la información NO está en el contexto, debe decir "No tengo información sobre esto en la documentación"

6. CHAT INTERACTIVO:
   - El sistema debe proporcionar una interfaz de chat interactiva por terminal
   - Debe permitir al usuario hacer preguntas de forma continua hasta que decida salir
   - Debe reconocer comandos de salida: 'salir', 'exit', 'quit' (case-insensitive)
   - Debe mostrar qué fragmentos de documentación se consultaron para cada respuesta

7. GESTIÓN DE VARIABLES DE ENTORNO:
   - El sistema debe cargar la API Key de OpenAI desde un archivo .env
   - Debe validar que la API Key existe antes de iniciar el sistema RAG
   - Debe mostrar mensajes de error claros si falta la configuración

8. MANEJO DE ERRORES:
   - El sistema debe manejar errores de carga de documento, creación de embeddings, y consultas al modelo
   - Debe mostrar mensajes de error amigables con sugerencias de solución
   - Debe permitir continuar el chat después de un error

REQUISITOS TÉCNICOS:

1. CONFIGURACIÓN DE TOKENIZERS:
   - El sistema debe configurar la variable de entorno TOKENIZERS_PARALLELISM="false" antes de importar cualquier módulo de HuggingFace
   - Esto evita warnings de paralelismo después de fork

2. IMPORTS REQUERIDOS:
   - Módulo os del sistema estándar de Python
   - Función load_dotenv del paquete dotenv
   - Clase ChatOpenAI del paquete langchain_openai
   - Clase HuggingFaceEmbeddings del paquete langchain_huggingface (CRÍTICO: NO usar langchain_community.embeddings que está deprecado)
   - Clase TextLoader del paquete langchain_community.document_loaders
   - Clase Chroma del paquete langchain_community.vectorstores
   - Clase RecursiveCharacterTextSplitter del paquete langchain_text_splitters
   - Clase ChatPromptTemplate del paquete langchain_core.prompts
   - Clase RunnablePassthrough del paquete langchain_core.runnables
   - Clase StrOutputParser del paquete langchain_core.output_parsers

3. CONFIGURACIÓN DE CHUNKS:
   - Tamaño de chunk: 500 caracteres
   - Solapamiento entre chunks: 50 caracteres
   - Función de longitud: len (contar caracteres)

4. CONFIGURACIÓN DE EMBEDDINGS:
   - Modelo: sentence-transformers/all-MiniLM-L6-v2
   - Dispositivo: CPU
   - Proveedor: HuggingFace (gratuito, sin costo)

5. CONFIGURACIÓN DEL MODELO LLM:
   - Temperature: 0.1 (muy baja para reducir alucinaciones)
   - max_tokens: 500
   - Modelo: Detectado automáticamente de la lista disponible

6. PROMPT TEMPLATE:
   - Debe instruir al modelo a responder ÚNICAMENTE usando la documentación proporcionada
   - Debe incluir instrucciones explícitas para evitar alucinaciones
   - Debe indicar que si no hay información en el contexto, debe decir claramente "No tengo información sobre esto en la documentación"

7. ESTRUCTURA DE ARCHIVOS:
   - Archivo de documentación: "documentacion_tecnica.md"
   - Directorio de base vectorial: "./chroma_db"
   - Archivo de configuración: ".env" (para API Key)

8. API MODERNA DE LANGCHAIN:
   - CRÍTICO: Usar langchain_huggingface para HuggingFaceEmbeddings (NO langchain_community.embeddings)
   - CRÍTICO: Usar retriever.invoke() para buscar documentos (NO métodos deprecados como get_relevant_documents)
   - CRÍTICO: Usar LangChain Expression Language (LCEL) con operador pipe para construir cadenas

9. MENSAJES DE USUARIO:
   - Todos los mensajes deben incluir emojis apropiados para mejor UX
   - Debe mostrar claramente el nombre del modelo en uso
   - Debe incluir separadores visuales entre conversaciones
   - Mensajes deben estar en español

10. DOCUMENTACIÓN:
    - Todas las funciones deben tener docstrings en español
    - Comentarios en español cuando sean necesarios
    - Código debe seguir convenciones PEP 8

RESULTADO ESPERADO:
El script debe demostrar claramente cómo el modelo AHORA SÍ puede responder sobre datos privados usando RAG, permitiendo al usuario hacer preguntas interactivamente y observando cómo el modelo responde usando el contexto de la documentación técnica, sin alucinar información que no está en la documentación.

FUNCIONALIDADES ESPERADAS:

1. FUNCIÓN DE DETECCIÓN DE MODELO:
   - El sistema debe implementar una función que detecte automáticamente qué modelo de OpenAI está disponible
   - Debe probar modelos en orden de preferencia: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo
   - Debe manejar errores cuando un modelo no está disponible y continuar con el siguiente
   - Debe retornar el nombre del modelo disponible o None si ninguno está disponible

2. FUNCIÓN DE CONFIGURACIÓN RAG:
   - El sistema debe implementar una función que configure el sistema RAG completo
   - La función debe realizar los siguientes pasos en orden:
     a) Cargar documento markdown desde archivo
     b) Dividir documento en fragmentos con tamaño y solapamiento especificados
     c) Crear embeddings vectoriales usando HuggingFace
     d) Almacenar vectores en ChromaDB persistente
     e) Detectar y configurar modelo LLM disponible
     f) Crear retriever que busque fragmentos relevantes
     g) Crear prompt template con instrucciones estrictas
     h) Construir cadena RAG usando LangChain Expression Language
   - La función debe mostrar mensajes informativos durante cada paso
   - La función debe retornar una tupla con (rag_chain, modelo, retriever)

3. FUNCIÓN PRINCIPAL:
   - El sistema debe implementar una función principal que orqueste todo el flujo
   - Debe validar la existencia de la API Key antes de iniciar
   - Debe configurar el sistema RAG llamando a la función de configuración
   - Debe mostrar mensajes de bienvenida e instrucciones al usuario
   - Debe implementar un loop interactivo que permita hacer preguntas continuamente
   - Debe procesar cada pregunta usando la cadena RAG
   - Debe mostrar la respuesta junto con información sobre los fragmentos consultados
   - Debe manejar errores de forma amigable y permitir continuar después de errores
   - Debe reconocer comandos de salida y terminar el programa apropiadamente

4. MENSAJES Y FEEDBACK AL USUARIO:
   - El sistema debe mostrar mensajes informativos durante la configuración del RAG
   - Debe mostrar mensajes de bienvenida con información sobre el sistema
   - Debe mostrar mensajes durante el procesamiento de cada pregunta
   - Debe mostrar claramente qué fragmentos de documentación se consultaron
   - Todos los mensajes deben incluir emojis apropiados para mejor UX
   - Mensajes deben estar en español

5. ESTRUCTURA Y ORGANIZACIÓN:
   - El código debe estar organizado en funciones claramente definidas
   - Cada función debe tener docstrings en español explicando su propósito
   - El código debe seguir convenciones PEP 8
   - Debe incluir punto de entrada estándar de Python
```

### ✅ Características Clave del Prompt

- **Especifica arquitectura completa**: todos los pasos del pipeline RAG
- **Detalla configuraciones**: valores específicos de chunk_size, modelo de embeddings, etc.
- **Incluye estructura LCEL**: muestra exactamente cómo construir la cadena
- **Define prompt template**: texto exacto del prompt para evitar alucinaciones
- **Especifica retornos**: qué debe retornar cada función

---

## 3. Script Híbrido (`main_hybrid.py`)

### 🎯 Objetivo
Crear un script que combine RAG con conocimiento del modelo: primero busca en documentación, si no encuentra usa conocimiento propio.

### 📝 Prompt Completo

```
Crea un script Python llamado `main_hybrid.py` que implemente un sistema HÍBRIDO que combine RAG con conocimiento del entrenamiento del modelo.

OBJETIVO DEL SISTEMA:
El script debe demostrar cómo combinar lo mejor de ambos mundos: datos privados mediante RAG y conocimiento general del modelo. El sistema debe implementar una estrategia híbrida que primero busca información en documentación privada usando RAG, y si no encuentra información relevante, usa el conocimiento del entrenamiento del modelo. Si el modelo tampoco sabe, debe decir explícitamente "No sé". El sistema debe mostrar claramente qué fuente utilizó para responder cada pregunta (documentación o conocimiento propio).

DEPENDENCIAS REQUERIDAS:
El proyecto utiliza Poetry como gestor de dependencias. Las dependencias necesarias son:
- Python 3.13 o superior
- langchain 1.0.7 o superior
- langchain-openai 1.0.3 o superior
- langchain-community 0.4.1 o superior
- langchain-huggingface 1.0.0 o superior (CRÍTICO: usar esta versión para evitar deprecaciones de HuggingFaceEmbeddings)
- chromadb 1.3.4 o superior
- python-dotenv 1.2.1 o superior
- openai 2.8.0 o superior
- sentence-transformers 5.1.2 o superior

REQUISITOS FUNCIONALES:

1. CONFIGURACIÓN DEL SISTEMA HÍBRIDO:
   - El sistema debe configurar un sistema RAG completo idéntico al script main.py (carga de documento, chunks, embeddings, vectorstore)
   - Debe configurar un retriever que busque los 3 fragmentos más relevantes
   - Debe detectar automáticamente qué modelo de OpenAI está disponible
   - Debe mostrar mensajes informativos durante la configuración indicando que es un sistema híbrido

2. EVALUACIÓN DE RELEVANCIA DE DOCUMENTOS:
   - El sistema debe implementar una función que evalúe si los documentos recuperados son relevantes para la pregunta del usuario
   - La evaluación debe considerar:
     - Si los documentos están vacíos o no existen → No relevante
     - Si el contenido total es menor a 50 caracteres → No relevante
     - Si el contenido total es mayor a 200 caracteres → Generalmente relevante (confiar en el retriever)
     - Si hay coincidencias de palabras clave importantes (palabras de más de 3 caracteres) entre la pregunta y el contenido → Relevante
     - Si el contenido es entre 50-200 caracteres y hay coincidencias de palabras clave (palabras de más de 2 caracteres) → Relevante
   - Debe filtrar stopwords en español antes de evaluar coincidencias
   - Debe retornar True si hay información relevante, False si no

3. ESTRATEGIA DE RESPUESTA HÍBRIDA:
   - El sistema debe implementar la siguiente estrategia de decisión:
     a) DETECCIÓN DE SOLICITUD EXPLÍCITA: Si el usuario pide explícitamente usar conocimiento fuera de las fuentes (palabras clave: 'fuera', 'fuentes', 'consultar', 'por fuera', 'sin documentación', 'conocimiento propio', 'entrenamiento'), debe usar directamente el conocimiento del modelo sin buscar en documentación
     b) BÚSQUEDA EN DOCUMENTACIÓN: Si no hay solicitud explícita, debe buscar documentos relevantes usando el retriever
     c) EVALUACIÓN DE RELEVANCIA: Debe evaluar si los documentos encontrados son relevantes usando la función de evaluación
     d) RESPUESTA CON RAG: Si hay información relevante, debe responder usando RAG con la documentación
     e) FALLBACK RAG DIRECTO: Si RAG responde con información insuficiente (menos de 50 caracteres y dice "no tengo información" o "no sé"), debe intentar una vez más con un prompt más directo usando los mismos documentos
     f) RESPUESTA CON CONOCIMIENTO PROPIO: Si no hay información relevante en la documentación, debe responder usando el conocimiento del entrenamiento del modelo
   - El sistema debe retornar tanto la respuesta como la fuente utilizada ("documentación" o "conocimiento del modelo")

4. MÚLTIPLES MODOS DE RESPUESTA:
   - El sistema debe implementar tres funciones de respuesta:
     a) responder_con_rag: Usa RAG cuando hay información en la documentación, con prompt estricto que instruye usar SOLO el contexto proporcionado
     b) responder_con_rag_directo: Usa RAG con documentos ya recuperados como fallback cuando el prompt normal falla, con un prompt más directo y simple
     c) responder_con_conocimiento_propio: Usa el conocimiento del entrenamiento del modelo cuando no hay información relevante en la documentación, con prompt que instruye ser honesto y decir "No sé" cuando no sabe

5. INDICADORES VISUALES DE FUENTE:
   - El sistema debe mostrar claramente qué fuente se utilizó para responder cada pregunta
   - Debe usar emojis diferentes: 📚 para documentación, 🧠 para conocimiento propio
   - Debe mostrar el texto "DOCUMENTACIÓN" o "CONOCIMIENTO PROPIO" junto al emoji
   - Si usó documentación, debe mostrar cuántos fragmentos se consultaron

6. CHAT INTERACTIVO:
   - El sistema debe proporcionar una interfaz de chat interactiva por terminal
   - Debe permitir al usuario hacer preguntas de forma continua hasta que decida salir
   - Debe reconocer comandos de salida: 'salir', 'exit', 'quit' (case-insensitive)
   - Debe mostrar mensajes informativos sobre la estrategia híbrida al inicio

7. GESTIÓN DE VARIABLES DE ENTORNO:
   - El sistema debe cargar la API Key de OpenAI desde un archivo .env
   - Debe validar que la API Key existe antes de iniciar el sistema
   - Debe mostrar mensajes de error claros si falta la configuración

8. MANEJO DE ERRORES:
   - El sistema debe manejar errores de configuración, carga de documento, y consultas al modelo
   - Debe mostrar mensajes de error amigables con sugerencias de solución
   - Debe permitir continuar el chat después de un error

REQUISITOS TÉCNICOS:

1. CONFIGURACIÓN DE TOKENIZERS:
   - El sistema debe configurar la variable de entorno TOKENIZERS_PARALLELISM="false" antes de importar cualquier módulo de HuggingFace
   - Esto evita warnings de paralelismo después de fork

2. IMPORTS REQUERIDOS:
   - Mismos imports que main.py: os, dotenv, langchain_openai, langchain_huggingface (CRÍTICO: NO usar langchain_community.embeddings), langchain_community.document_loaders, langchain_community.vectorstores, langchain_text_splitters, langchain_core.prompts, langchain_core.runnables, langchain_core.output_parsers

3. CONSTANTES DE CONFIGURACIÓN:
   - DOCUMENTO: "documentacion_tecnica.md"
   - CHROMA_DB_DIR: "./chroma_db"

4. CONFIGURACIÓN RAG (igual que main.py):
   - Tamaño de chunk: 500 caracteres
   - Solapamiento: 50 caracteres
   - Modelo de embeddings: sentence-transformers/all-MiniLM-L6-v2
   - Dispositivo: CPU
   - Retriever: busca 3 fragmentos más relevantes

5. CONFIGURACIÓN DEL MODELO LLM:
   - Temperature: 0.1 (muy baja para reducir alucinaciones)
   - max_tokens: 500
   - Modelo: Detectado automáticamente

6. EVALUACIÓN DE RELEVANCIA - STOPWORDS EN ESPAÑOL:
   - El sistema debe filtrar estos stopwords exactos antes de evaluar coincidencias: 'qué', 'cómo', 'cuándo', 'dónde', 'por', 'para', 'con', 'de', 'la', 'el', 'un', 'una', 'es', 'son', 'está', 'están', 'sobre', 'los', 'las', 'y', 'o', 'pero', 'si', 'no', 'en', 'a', 'que', 'se', 'le', 'te', 'me', 'nos', 'les', 'puedes', 'puede', 'puedo', 'pueden', 'pueda', 'puedan', 'sirve', 'sirven', 'consultar', 'fuera', 'fuentes', 'tus', 'sus', 'mis', 'nuestros', 'vuestros', 'trata', 'como', 'instala', 'instalar'

7. PROMPT TEMPLATES:
   - Template RAG: Debe instruir usar ÚNICAMENTE la documentación proporcionada, ser exhaustivo si hay información, y decir "No tengo información sobre esto en la documentación" si no hay información
   - Template RAG Directo: Debe ser más simple y directo, instruyendo usar SOLO la información de la documentación
   - Template Conocimiento Propio: Debe instruir usar conocimiento general, ser honesto, y decir "No sé sobre..." cuando no sabe

8. DETECCIÓN DE SOLICITUD EXPLÍCITA:
   - El sistema debe detectar si el usuario pide explícitamente usar conocimiento fuera de fuentes
   - Palabras clave a detectar: 'fuera', 'fuentes', 'consultar', 'por fuera', 'sin documentación', 'conocimiento propio', 'entrenamiento'
   - Si detecta estas palabras, debe limpiar la pregunta removiendo estas palabras y usar conocimiento propio directamente

9. API MODERNA DE LANGCHAIN:
   - CRÍTICO: Usar langchain_huggingface para HuggingFaceEmbeddings (NO langchain_community.embeddings)
   - CRÍTICO: Usar retriever.invoke() para buscar documentos (NO métodos deprecados)
   - CRÍTICO: Usar LangChain Expression Language (LCEL) con operador pipe para construir cadenas

10. MENSAJES DE USUARIO:
    - Todos los mensajes deben incluir emojis apropiados
    - Debe mostrar claramente qué fuente se usó (📚 DOCUMENTACIÓN o 🧠 CONOCIMIENTO PROPIO)
    - Debe incluir separadores visuales entre conversaciones
    - Mensajes deben estar en español

11. DOCUMENTACIÓN:
    - Todas las funciones deben tener docstrings en español explicando su propósito
    - Comentarios en español cuando sean necesarios
    - Código debe seguir convenciones PEP 8

RESULTADO ESPERADO:
El script debe demostrar cómo combinar lo mejor de ambos mundos: datos privados mediante RAG y conocimiento general del modelo. El sistema debe decidir inteligentemente qué fuente usar para responder cada pregunta, mostrando claramente al usuario si la respuesta viene de la documentación o del conocimiento propio del modelo. El usuario debe poder hacer preguntas interactivamente y observar cómo el sistema evalúa la relevancia y selecciona la mejor fuente de información.
```

### ✅ Características Clave del Prompt

- **Define estrategia completa**: flujo de decisión paso a paso
- **Especifica múltiples funciones**: cada una con propósito claro
- **Detalla lógica de evaluación**: cómo determinar relevancia
- **Incluye casos especiales**: detección de solicitud explícita, fallback
- **Define estructura completa**: todas las funciones necesarias

---

## 🎓 Mejores Prácticas de Prompt Engineering

### 1. **Estructura Clara**
- Separar en secciones: REQUISITOS FUNCIONALES, REQUISITOS TÉCNICOS, ESTRUCTURA ESPERADA
- Usar listas numeradas o con viñetas para mejor legibilidad

### 2. **Especificidad**
- Incluir valores exactos: `temperature=0.1`, `chunk_size=500`
- Mencionar librerías específicas: `langchain_openai`, `HuggingFaceEmbeddings`
- Definir nombres de funciones y variables esperadas

### 3. **Contexto y Objetivo**
- Empezar con el objetivo educativo o funcional
- Explicar el "por qué" además del "qué"

### 4. **Descripciones Detalladas Sin Código**
- Describir exactamente qué debe hacer cada función sin mostrar código
- Especificar valores exactos, parámetros, y estructuras de datos esperadas
- Usar lenguaje descriptivo que permita generar código preciso

### 5. **Restricciones y Validaciones**
- Especificar qué NO debe hacer (evitar alucinaciones)
- Definir manejo de errores esperado

### 6. **Documentación Esperada**
- Solicitar docstrings en español
- Pedir mensajes informativos para UX

---

## 📊 Comparación de Prompts

| Aspecto | Script Sin Contexto | Script con RAG | Script Híbrido |
|---------|-------------------|----------------|----------------|
| **Complejidad** | Baja | Media | Alta |
| **Librerías** | 2 (openai, dotenv) | 7+ (LangChain completo) | 7+ (LangChain completo) |
| **Funciones** | 2 | 3 | 8+ |
| **Lógica de Decisión** | Ninguna | Simple (solo RAG) | Compleja (RAG + evaluación + fallback) |
| **Prompt Length** | ~300 palabras | ~500 palabras | ~800 palabras |

---

## 🚀 Uso de los Prompts

### Opción 1: Copiar y Pegar Directo
Copia el prompt completo en tu herramienta de IA favorita (Claude, GPT-4, etc.) y solicita la generación del código.

### Opción 2: Adaptación Incremental
1. Empieza con el prompt base
2. Agrega requisitos específicos de tu proyecto
3. Refina según necesidades

### Opción 3: Prompt Modular
Divide el prompt en secciones y genera cada parte por separado, luego combina.

---

## ⚠️ Errores Comunes y Cómo Evitarlos

### 1. **Error: `'VectorStoreRetriever' object has no attribute 'get_relevant_documents'`**

**Causa**: Uso de métodos deprecados de LangChain.

**Solución en el prompt**:
- Especificar claramente en el prompt que se debe usar el método invoke del retriever pasando la pregunta como argumento para buscar documentos
- NO mencionar en el prompt métodos como get_relevant_documents o retrieve que están deprecados
- Especificar que en LCEL, el retriever se usa directamente con el operador pipe, donde el operador pipe ejecuta automáticamente invoke
- Describir que para obtener fragmentos consultados se debe llamar al método invoke del retriever pasando la pregunta, lo cual retorna una lista de documentos

### 2. **Warning: `HuggingFaceEmbeddings` was deprecated**

**Causa**: Uso de import deprecado desde `langchain_community.embeddings`.

**Solución en el prompt**:
- Especificar explícitamente en el prompt que se debe importar HuggingFaceEmbeddings desde el paquete langchain_huggingface
- NO mencionar en el prompt importar desde langchain_community.embeddings que está deprecado
- Incluir en las dependencias la librería langchain-huggingface versión 1.0.0 o superior
- Agregar una nota crítica en el prompt indicando que usar langchain_community.embeddings está deprecado y causará warnings

### 3. **Dependencia faltante: `langchain-huggingface`**

**Solución en el prompt**:
- Incluir en la lista de dependencias: "langchain-huggingface (requerida para HuggingFaceEmbeddings)"
- Mencionar que debe instalarse: `poetry add langchain-huggingface` o `pip install langchain-huggingface`

### 4. **Timeout al descargar modelos de HuggingFace**

**Solución en el prompt**:
- Mencionar que la primera descarga puede tomar tiempo
- Sugerir manejo de errores con mensajes informativos
- Opcional: mencionar que el modelo se cachea localmente después de la primera descarga

---

## 💡 Tips Adicionales

1. **Iteración**: Los prompts pueden necesitar refinamiento. Prueba y ajusta.
2. **Especificidad**: Mientras más específico, mejor resultado. Incluye valores exactos.
3. **Ejemplos**: Si tienes código de referencia, inclúyelo en el prompt.
4. **Validación**: Siempre prueba el código generado antes de usarlo en producción.
5. **Documentación**: Solicita explícitamente documentación en español si la necesitas.
6. **API Moderna**: Siempre especifica usar la API moderna de LangChain (v1.0+), NO métodos deprecados.
7. **Dependencias**: Lista todas las dependencias necesarias, incluyendo `langchain-huggingface`.

---

## 📦 Dependencias Requeridas

Para que los scripts generados funcionen correctamente, asegúrate de tener estas dependencias en `pyproject.toml`:

```toml
[tool.poetry.dependencies]
python = "^3.13"
langchain = "^1.0.7"
langchain-openai = "^1.0.3"
langchain-community = "^0.4.1"
langchain-huggingface = "^1.0.0"  # ⚠️ REQUERIDA para HuggingFaceEmbeddings
chromadb = "^1.3.4"
python-dotenv = "^1.2.1"
openai = "^2.8.0"
sentence-transformers = "^5.1.2"
```

**Instalación**:
```bash
poetry add langchain-huggingface
# o
pip install langchain-huggingface
```

---

## 📝 Notas Finales

Estos prompts están diseñados para generar código funcional y bien documentado. Sin embargo, siempre:

- ✅ Revisa el código generado
- ✅ Prueba la funcionalidad
- ✅ Ajusta según tus necesidades específicas
- ✅ Valida dependencias y configuraciones
- ✅ Verifica que uses la API moderna de LangChain (v1.0+)
- ✅ Asegúrate de tener `langchain-huggingface` instalado

Los prompts pueden adaptarse para otros frameworks o lenguajes cambiando las librerías y estructuras mencionadas.

