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

REQUISITOS FUNCIONALES:
1. Chat interactivo con OpenAI GPT usando la API oficial
2. Detección automática del modelo disponible (gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo)
3. Configuración anti-alucinación:
   - Temperature: 0.1 (muy baja)
   - top_p: 0.9
   - System prompt que instruya al modelo a decir "No sé sobre..." cuando no tiene información
4. Loop de conversación interactivo con comando 'salir' para terminar
5. Manejo de errores y validación de API Key desde archivo .env

REQUISITOS TÉCNICOS:
- Usar `openai` (cliente oficial) y `python-dotenv`
- Función `detectar_modelo(client)` que pruebe modelos en orden de preferencia
- Función `main()` con loop interactivo
- System prompt que evite alucinaciones: "Eres un asistente honesto y directo. Si no conoces o no tienes información sobre algo, debes decir claramente 'No sé sobre...' o 'No tengo información sobre...' en lugar de inventar o suponer."
- Documentación en español en docstrings
- Mensajes informativos con emojis para mejor UX

ESTRUCTURA ESPERADA:
- Imports: os, OpenAI, load_dotenv
- Función detectar_modelo(client) → retorna modelo disponible o None
- Función main() → configuración, loop interactivo, manejo de errores
- Punto de entrada: if __name__ == "__main__"

El script debe ser simple, directo y demostrar claramente que el modelo NO conoce datos privados.
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

REQUISITOS FUNCIONALES:
1. Sistema RAG completo con los siguientes pasos:
   - Cargar documento markdown desde archivo "documentacion_tecnica.md"
   - Dividir documento en chunks (chunk_size=500, chunk_overlap=50)
   - Crear embeddings usando HuggingFaceEmbeddings (modelo: sentence-transformers/all-MiniLM-L6-v2)
   - Almacenar vectores en ChromaDB (directorio: ./chroma_db)
   - Configurar retriever que busque los 3 fragmentos más relevantes
   - Crear cadena RAG usando LangChain Expression Language (LCEL)
2. Detección automática del modelo OpenAI disponible
3. Configuración anti-alucinación:
   - Temperature: 0.1
   - Prompt template estricto que instruya usar SOLO el contexto proporcionado
4. Chat interactivo que muestre cuántos fragmentos se consultaron
5. Manejo de errores y validación de API Key

REQUISITOS TÉCNICOS:
- Librerías: langchain_openai, langchain_community (embeddings, document_loaders, vectorstores), langchain_text_splitters, langchain_core (prompts, runnables, output_parsers), python-dotenv
- Función `detectar_modelo()` que pruebe modelos en orden
- Función `configurar_rag()` que retorne (rag_chain, modelo, retriever)
- Función `main()` con loop interactivo
- Constantes de configuración: DOCUMENTO = "documentacion_tecnica.md", CHROMA_DB_DIR = "./chroma_db"
- Prompt template que diga: "Responde SOLO usando la información que está en el contexto proporcionado. Si la información NO está en el contexto, di claramente 'No tengo información sobre esto en la documentación'"
- Documentación en español en docstrings
- Mensajes informativos durante la configuración

ESTRUCTURA ESPERADA:
- Imports de LangChain
- Constantes de configuración
- Función detectar_modelo() → retorna modelo o None
- Función configurar_rag() → configura todo el sistema RAG, retorna (rag_chain, modelo, retriever)
- Función format_docs(docs) → formatea documentos recuperados
- Función main() → configura RAG y ejecuta chat interactivo
- Punto de entrada: if __name__ == "__main__"

CADENA RAG (LCEL):
Usar LangChain Expression Language:
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

El script debe demostrar claramente cómo el modelo AHORA SÍ puede responder sobre datos privados usando RAG.
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

REQUISITOS FUNCIONALES:
1. Sistema híbrido con estrategia de decisión:
   - PRIMERO: Buscar en documentación usando RAG
   - EVALUAR: Determinar si los documentos encontrados son relevantes
   - SI ES RELEVANTE: Usar RAG con documentación
   - SI NO ES RELEVANTE: Usar conocimiento propio del modelo
   - DETECCIÓN ESPECIAL: Si el usuario pide explícitamente usar conocimiento fuera de fuentes, hacerlo directamente
2. Configuración RAG completa (igual que main.py):
   - Cargar documento, dividir en chunks, crear embeddings HuggingFace, almacenar en ChromaDB
   - Retriever con k=3 fragmentos
3. Función de evaluación de relevancia:
   - Analizar si documentos recuperados contienen información relevante
   - Verificar contenido sustancial (>200 caracteres) y coincidencias de palabras clave
   - Retornar True/False según relevancia
4. Dos modos de respuesta:
   - responder_con_rag(): Usa RAG cuando hay información en documentación
   - responder_con_conocimiento_propio(): Usa conocimiento del modelo cuando no hay información relevante
5. Indicadores visuales: Mostrar qué fuente se usó (📚 DOCUMENTACIÓN o 🧠 CONOCIMIENTO PROPIO)
6. Chat interactivo con información sobre fragmentos consultados

REQUISITOS TÉCNICOS:
- Mismas librerías que main.py (LangChain completo)
- Función `configurar_sistema_hibrido()` → retorna (retriever, llm, modelo, vectorstore)
- Función `evaluar_relevancia_documentos(docs, pregunta)` → retorna bool
- Función `responder_con_rag(pregunta, retriever, llm)` → retorna str
- Función `responder_con_conocimiento_propio(pregunta, llm)` → retorna str
- Función `responder_hibrido(pregunta, retriever, llm)` → retorna (respuesta, fuente)
- Función `responder_con_rag_directo(pregunta, docs, llm)` → fallback cuando RAG falla pero hay docs relevantes
- Detección de solicitud explícita: palabras clave ['fuera', 'fuentes', 'consultar', 'por fuera', 'sin documentación']
- Configuración anti-alucinación: temperature=0.1
- Prompts específicos:
  - RAG: "DEBES responder usando la información del contexto proporcionado. NO uses conocimiento fuera del contexto"
  - Conocimiento propio: "Responde usando tu conocimiento general. Si NO sabes, di 'No sé sobre...'"

ESTRATEGIA DE EVALUACIÓN DE RELEVANCIA:
- Si contenido_total > 200 caracteres Y hay palabras clave importantes en el contenido → relevante
- Si contenido_total >= 50 Y hay coincidencias de palabras clave → relevante
- Filtrar stopwords en español antes de evaluar palabras clave
- Confiar en el retriever cuando encuentra contenido sustancial

ESTRUCTURA ESPERADA:
- Imports completos de LangChain
- Constantes: DOCUMENTO, CHROMA_DB_DIR
- detectar_modelo()
- configurar_sistema_hibrido()
- evaluar_relevancia_documentos(docs, pregunta)
- responder_con_rag(pregunta, retriever, llm)
- responder_con_rag_directo(pregunta, docs, llm) [fallback]
- responder_con_conocimiento_propio(pregunta, llm)
- responder_hibrido(pregunta, retriever, llm) [función principal de decisión]
- main()
- Punto de entrada

LÓGICA DE responder_hibrido():
1. Detectar si usuario pide explícitamente conocimiento fuera → usar conocimiento propio directamente
2. Buscar documentos con retriever
3. Evaluar relevancia con evaluar_relevancia_documentos()
4. Si relevante → usar RAG
5. Si RAG dice "no tengo información" pero hay docs relevantes → intentar responder_con_rag_directo()
6. Si no relevante → usar conocimiento propio
7. Retornar (respuesta, fuente) donde fuente es "documentación" o "conocimiento del modelo"

El script debe demostrar cómo combinar lo mejor de ambos mundos: datos privados mediante RAG y conocimiento general del modelo.
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

### 4. **Ejemplos Concretos**
- Incluir código de ejemplo cuando sea relevante (como la cadena LCEL)
- Mostrar estructuras esperadas

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

## 💡 Tips Adicionales

1. **Iteración**: Los prompts pueden necesitar refinamiento. Prueba y ajusta.
2. **Especificidad**: Mientras más específico, mejor resultado. Incluye valores exactos.
3. **Ejemplos**: Si tienes código de referencia, inclúyelo en el prompt.
4. **Validación**: Siempre prueba el código generado antes de usarlo en producción.
5. **Documentación**: Solicita explícitamente documentación en español si la necesitas.

---

## 📝 Notas Finales

Estos prompts están diseñados para generar código funcional y bien documentado. Sin embargo, siempre:

- ✅ Revisa el código generado
- ✅ Prueba la funcionalidad
- ✅ Ajusta según tus necesidades específicas
- ✅ Valida dependencias y configuraciones

Los prompts pueden adaptarse para otros frameworks o lenguajes cambiando las librerías y estructuras mencionadas.

