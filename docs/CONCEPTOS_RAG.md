# 📚 Conceptos Fundamentales de RAG - Guía para Masterclass

Este documento explica los conceptos clave de RAG (Retrieval Augmented Generation) de manera clara y didáctica.

---

## 📋 Índice

1. [Chunks (Fragmentos)](#1-chunks-fragmentos)
2. [Embeddings (Vectores)](#2-embeddings-vectores)
3. [ChromaDB (Base de Datos Vectorial)](#3-chromadb-base-de-datos-vectorial)
4. [Retriever (Recuperador)](#4-retriever-recuperador)
5. [Cadena RAG y LCEL](#5-cadena-rag-y-lcel)

---

## 1. Chunks (Fragmentos)

### ¿Qué son los Chunks?

Los **chunks** (fragmentos) son **porciones pequeñas de texto** en las que se divide un documento grande para facilitar su procesamiento y recuperación.

### ¿Cuál es su uso?

**Problema sin chunks:**
- Un documento de 10,000 palabras es difícil de procesar
- Buscar información específica requiere leer todo el documento
- Los modelos tienen límites de tokens (contexto limitado)

**Solución con chunks:**
- Dividir el documento en fragmentos de 500-1000 palabras
- Cada chunk puede procesarse independientemente
- Buscar solo los chunks relevantes para una pregunta específica
- Enviar solo los chunks relevantes al modelo (ahorra tokens y costos)

### ¿Cómo se configuran?

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # Tamaño máximo de cada chunk (en caracteres)
    chunk_overlap=50,      # Caracteres que se solapan entre chunks
    length_function=len    # Función para medir longitud
)

chunks = text_splitter.split_documents(documentos)
```

**Parámetros importantes:**

| Parámetro | Descripción | Ejemplo | Efecto |
|-----------|-------------|---------|--------|
| `chunk_size` | Tamaño máximo del fragmento | 500 caracteres | Chunks más grandes = más contexto, pero menos precisión |
| `chunk_overlap` | Solapamiento entre chunks | 50 caracteres | Evita perder información en los bordes |
| `length_function` | Cómo medir longitud | `len` | Puede usar tokens, palabras, etc. |

**¿Por qué overlap (solapamiento)?**

```
Documento original:
"HenryPy es una librería de Python. Para instalarla usa: pip install henrypy"

Sin overlap:
Chunk 1: "HenryPy es una librería de Python."
Chunk 2: "Para instalarla usa: pip install henrypy"

Con overlap (50 chars):
Chunk 1: "HenryPy es una librería de Python. Para instalarla"
Chunk 2: "una librería de Python. Para instalarla usa: pip install henrypy"
```

El overlap asegura que información importante en los bordes no se pierda.

---

## 2. Embeddings (Vectores)

### ¿Qué son los Embeddings?

Los **embeddings** son **representaciones numéricas (vectores) de texto** que capturan el significado semántico. Son arrays de números que representan el "sentido" del texto.

### ¿Cuál es su uso?

**Problema:**
- Las computadoras no entienden texto directamente
- Necesitamos una forma de comparar textos por significado, no por palabras exactas

**Solución con embeddings:**
- Convertir texto → vector de números (ej: [0.2, -0.5, 0.8, ...])
- Textos similares tienen vectores similares
- Podemos buscar textos similares usando matemáticas (distancia entre vectores)

**Ejemplo visual:**

```
"Instalar Python"     → [0.2, -0.3, 0.5, 0.1, ...]
"Instalar librería"    → [0.25, -0.28, 0.48, 0.12, ...]  ← Similar (cerca en espacio)
"Comer pizza"          → [-0.8, 0.4, -0.2, 0.9, ...]      ← Diferente (lejos en espacio)
```

### ¿Cómo se configuran?

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}  # o 'cuda' para GPU
)

# Convertir texto a vector
vector = embeddings.embed_query("¿Cómo instalo HenryPy?")
# Resultado: [0.123, -0.456, 0.789, ...] (vector de 384 números)
```

**Modelos comunes:**

| Modelo | Tamaño | Dimensiones | Uso |
|--------|--------|-------------|-----|
| `all-MiniLM-L6-v2` | ~80MB | 384 | Rápido, bueno para general |
| `all-mpnet-base-v2` | ~420MB | 768 | Más preciso, más lento |
| `text-embedding-ada-002` (OpenAI) | API | 1536 | Muy preciso, requiere API key |

**En nuestro proyecto:**
- Usamos `all-MiniLM-L6-v2` porque es **gratis** y funciona localmente
- Genera vectores de 384 dimensiones
- No requiere conexión a internet después de la primera descarga

### ¿Qué relación tienen con los Chunks?

**Flujo completo:**

```
1. Documento original
   ↓
2. Dividir en CHUNKS (fragmentos de texto)
   ↓
3. Convertir cada CHUNK a EMBEDDING (vector numérico)
   ↓
4. Almacenar embeddings en base vectorial (ChromaDB)
   ↓
5. Cuando llega una pregunta:
   - Convertir pregunta a embedding
   - Buscar chunks con embeddings similares
   - Recuperar los chunks más relevantes
```

**Ejemplo práctico:**

```python
# 1. Documento original
documento = "HenryPy es una librería de Python..."

# 2. Dividir en chunks
chunks = [
    "HenryPy es una librería de Python diseñada para...",
    "Para instalar HenryPy usa: pip install henrypy...",
    "La configuración requiere una API_KEY..."
]

# 3. Convertir cada chunk a embedding
embeddings_chunk1 = [0.1, -0.2, 0.3, ...]  # Vector de 384 números
embeddings_chunk2 = [0.15, -0.18, 0.28, ...]
embeddings_chunk3 = [0.05, -0.25, 0.35, ...]

# 4. Pregunta del usuario
pregunta = "¿Cómo instalo HenryPy?"
embedding_pregunta = [0.14, -0.19, 0.29, ...]  # Similar a chunk2

# 5. Buscar chunk más similar (usando distancia matemática)
# Resultado: chunk2 es el más relevante
```

---

## 3. ChromaDB (Base de Datos Vectorial)

### ¿Qué es ChromaDB?

**ChromaDB** es una **base de datos especializada** para almacenar y buscar **vectores (embeddings)** de manera eficiente.

### ¿Qué tiene de particular?

**Características principales:**

1. **Búsqueda por similitud semántica**: Encuentra textos similares, no exactos
2. **Optimizada para vectores**: Diseñada específicamente para operaciones con embeddings
3. **Búsqueda rápida**: Usa algoritmos especializados (como LSH, HNSW) para búsqueda rápida
4. **Persistencia**: Guarda los datos en disco (no se pierden al cerrar)
5. **Ligera**: No requiere servidor, funciona como biblioteca Python

### ¿Por qué no usar SQLite para la misma tarea?

**SQLite vs ChromaDB:**

| Aspecto | SQLite | ChromaDB |
|---------|--------|----------|
| **Tipo de búsqueda** | Exacta (WHERE texto = "X") | Por similitud semántica |
| **Optimización** | Para texto exacto | Para vectores numéricos |
| **Búsqueda semántica** | ❌ No nativa | ✅ Nativa y rápida |
| **Ejemplo** | Buscar "instalar" encuentra solo "instalar" | Buscar "instalar" encuentra "instalación", "setup", "configurar" |

**Ejemplo práctico:**

**Con SQLite:**
```sql
-- Solo encuentra coincidencias exactas
SELECT * FROM documentos WHERE texto LIKE '%instalar%';
-- No encuentra: "instalación", "setup", "configurar"
```

**Con ChromaDB:**
```python
# Encuentra textos semánticamente similares
results = vectorstore.similarity_search("instalar")
# Encuentra: "instalación", "setup", "configurar", "instalar"
```

**¿Cuándo usar cada uno?**

- **SQLite**: Datos estructurados, búsquedas exactas, relaciones complejas
- **ChromaDB**: Búsqueda semántica, RAG, recomendaciones, búsqueda por significado

**En nuestro proyecto:**
- Usamos ChromaDB porque necesitamos **búsqueda semántica**
- Cuando preguntamos "instalar HenryPy", queremos encontrar chunks sobre "instalación", "setup", etc.
- SQLite no puede hacer esto eficientemente

---

## 4. Retriever (Recuperador)

### ¿Qué es un Retriever?

Un **retriever** es un componente que **busca y recupera los chunks más relevantes** de la base vectorial para una pregunta específica.

### ¿Cuál es su utilidad en un RAG?

**Función principal:**
1. Recibe una pregunta del usuario
2. Convierte la pregunta a embedding
3. Busca en ChromaDB los chunks con embeddings más similares
4. Retorna los N chunks más relevantes

**Sin retriever:**
- Tendríamos que buscar manualmente en todos los chunks
- No sabríamos cuáles son relevantes
- Enviaríamos información irrelevante al modelo (desperdicio de tokens)

**Con retriever:**
- Encuentra automáticamente los chunks relevantes
- Solo envía información útil al modelo
- Ahorra tokens y mejora la precisión

### ¿Qué son los fragmentos?

Los **fragmentos** son los **chunks recuperados** por el retriever. Son las porciones de texto que el retriever considera más relevantes para responder la pregunta.

**Ejemplo:**

```python
# Pregunta del usuario
pregunta = "¿Cómo instalo HenryPy?"

# Retriever busca y encuentra los 3 fragmentos más relevantes
fragmentos = retriever.invoke(pregunta)
# Resultado:
# [
#   "Para instalar HenryPy, usa: pip install henrypy...",
#   "La instalación estándar se realiza a través de pip...",
#   "Si olvidas este paso, las funciones fallarán..."
# ]

# Estos fragmentos se envían al modelo como contexto
```

**Configuración del retriever:**

```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}  # Retorna los 3 fragmentos más relevantes
)
```

**Parámetro `k`:**
- `k=3`: Retorna los 3 fragmentos más relevantes
- `k=5`: Retorna los 5 fragmentos más relevantes
- Más fragmentos = más contexto, pero también más tokens y posiblemente información menos relevante

---

## 5. Cadena RAG y LCEL

### ¿Qué es una Cadena RAG?

Una **cadena RAG** es el **flujo completo** que conecta todos los componentes: pregunta → búsqueda → contexto → respuesta.

**Componentes de la cadena:**

```
1. PREGUNTA del usuario
   ↓
2. RETRIEVER busca chunks relevantes
   ↓
3. FORMAT_DOCS formatea los chunks
   ↓
4. PROMPT template combina contexto + pregunta
   ↓
5. LLM genera respuesta usando el contexto
   ↓
6. OUTPUT_PARSER formatea la respuesta
   ↓
7. RESPUESTA final
```

### ¿Qué es LCEL?

**LCEL** (LangChain Expression Language) es una forma **declarativa y funcional** de construir cadenas en LangChain usando el operador `|` (pipe).

**Sintaxis tradicional (sin LCEL):**
```python
# Complicado y verboso
docs = retriever.invoke(pregunta)
formatted_docs = format_docs(docs)
prompt_input = {"context": formatted_docs, "question": pregunta}
prompt_output = prompt.invoke(prompt_input)
llm_output = llm.invoke(prompt_output)
respuesta = output_parser.invoke(llm_output)
```

**Sintaxis LCEL (moderna):**
```python
# Simple y elegante
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

respuesta = rag_chain.invoke(pregunta)
```

### ¿Cómo funciona LCEL?

**Operador `|` (pipe):**
- Similar a pipes en terminal: `cat file.txt | grep "texto" | head -5`
- Cada componente se ejecuta secuencialmente
- La salida de uno es la entrada del siguiente

**Componentes LCEL:**

```python
rag_chain = (
    # PASO 1: Preparar inputs
    {
        "context": retriever | format_docs,  # Buscar y formatear chunks
        "question": RunnablePassthrough()     # Pasar pregunta sin modificar
    }
    |  # ↓
    # PASO 2: Crear prompt
    prompt  # Combina contexto + pregunta en un prompt
    |  # ↓
    # PASO 3: Generar respuesta
    llm  # Modelo de lenguaje genera respuesta
    |  # ↓
    # PASO 4: Formatear salida
    StrOutputParser()  # Convierte a string simple
)
```

**Ventajas de LCEL:**

1. **Legible**: Se lee de izquierda a derecha como un flujo
2. **Composable**: Fácil agregar/quitar componentes
3. **Eficiente**: LangChain optimiza la ejecución
4. **Moderno**: Es la forma recomendada en LangChain v1.0+

**Ejemplo paso a paso:**

```python
# Input
pregunta = "¿Cómo instalo HenryPy?"

# Paso 1: Retriever busca chunks
chunks = retriever.invoke(pregunta)
# Resultado: [chunk1, chunk2, chunk3]

# Paso 2: Formatear chunks
contexto = format_docs(chunks)
# Resultado: "chunk1\n\nchunk2\n\nchunk3"

# Paso 3: Crear prompt
prompt_completo = prompt.format(context=contexto, question=pregunta)
# Resultado: "Eres un asistente...\nContexto: chunk1...\nPregunta: ¿Cómo instalo HenryPy?"

# Paso 4: LLM genera respuesta
respuesta_llm = llm.invoke(prompt_completo)
# Resultado: Objeto Message con contenido

# Paso 5: Parsear a string
respuesta_final = StrOutputParser().invoke(respuesta_llm)
# Resultado: "Para instalar HenryPy, usa: pip install henrypy..."
```

**Con LCEL, todo esto se hace automáticamente:**

```python
respuesta = rag_chain.invoke(pregunta)
# ¡Listo! Todo el flujo se ejecuta automáticamente
```

---

## 🔄 Flujo Completo de RAG (Resumen Visual)

```
┌─────────────────────────────────────────────────────────────┐
│                    DOCUMENTO ORIGINAL                        │
│         "documentacion_tecnica.md" (texto plano)            │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ↓
        ┌───────────────────────────────┐
        │   TEXT SPLITTER (Chunks)      │
        │   Divide en fragmentos de 500  │
        │   caracteres con overlap 50   │
        └───────────────┬───────────────┘
                        │
                        ↓
        ┌───────────────────────────────┐
        │   EMBEDDINGS                  │
        │   Convierte cada chunk a      │
        │   vector numérico (384 dim)   │
        └───────────────┬───────────────┘
                        │
                        ↓
        ┌───────────────────────────────┐
        │   CHROMADB                    │
        │   Almacena vectores para      │
        │   búsqueda rápida             │
        └───────────────┬───────────────┘
                        │
                        ↓ (Cuando llega pregunta)
        ┌───────────────────────────────┐
        │   RETRIEVER                   │
        │   Busca chunks más relevantes │
        │   (k=3 fragmentos)           │
        └───────────────┬───────────────┘
                        │
                        ↓
        ┌───────────────────────────────┐
        │   CADENA RAG (LCEL)           │
        │   retriever → format → prompt │
        │   → llm → parser              │
        └───────────────┬───────────────┘
                        │
                        ↓
        ┌───────────────────────────────┐
        │   RESPUESTA FINAL             │
        │   Con contexto de documentación│
        └───────────────────────────────┘
```

---

## 📝 Resumen de Conceptos Clave

| Concepto | ¿Qué es? | ¿Para qué sirve? | Ejemplo |
|----------|----------|------------------|---------|
| **Chunk** | Fragmento de texto | Dividir documentos grandes | "Para instalar HenryPy usa pip..." |
| **Embedding** | Vector numérico | Representar significado del texto | [0.2, -0.3, 0.5, ...] |
| **ChromaDB** | Base de datos vectorial | Buscar por similitud semántica | Encuentra textos similares |
| **Retriever** | Componente de búsqueda | Encontrar chunks relevantes | Retorna los 3 mejores chunks |
| **Cadena RAG** | Flujo completo | Conectar todos los componentes | pregunta → búsqueda → respuesta |
| **LCEL** | Lenguaje de expresiones | Construir cadenas de forma elegante | `retriever \| prompt \| llm` |

---

## 🎯 Preguntas Frecuentes

### ¿Por qué dividir en chunks si puedo enviar todo el documento?

- **Límites de tokens**: Los modelos tienen límites (ej: 4K, 8K, 32K tokens)
- **Costo**: Más tokens = más costo
- **Precisión**: Solo enviar información relevante mejora la respuesta
- **Velocidad**: Procesar chunks pequeños es más rápido

### ¿Qué tamaño de chunk es mejor?

- **Chunks pequeños (200-300)**: Más precisos, pero pueden perder contexto
- **Chunks medianos (500-800)**: Balance entre precisión y contexto (recomendado)
- **Chunks grandes (1000+)**: Más contexto, pero menos precisión

**Recomendación**: Empieza con 500 y ajusta según resultados.

### ¿Por qué usar embeddings locales (HuggingFace) vs API (OpenAI)?

| Aspecto | HuggingFace (Local) | OpenAI (API) |
|---------|---------------------|--------------|
| **Costo** | ✅ Gratis | ❌ Pago por uso |
| **Velocidad** | ⚠️ Primera vez lenta (descarga) | ✅ Rápido |
| **Privacidad** | ✅ 100% local | ⚠️ Envía datos a API |
| **Precisión** | ⚠️ Buena | ✅ Excelente |

**Recomendación**: Para desarrollo y demos, usa HuggingFace. Para producción, considera OpenAI.

### ¿Cuántos fragmentos (k) debo recuperar?

- **k=1**: Muy específico, puede perder contexto
- **k=3**: Balance recomendado (usado en nuestro proyecto)
- **k=5**: Más contexto, pero puede incluir información menos relevante
- **k=10+**: Demasiado contexto, puede confundir al modelo

**Recomendación**: Empieza con k=3 y ajusta según resultados.

---

## 📚 Recursos Adicionales

- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [HuggingFace Sentence Transformers](https://www.sbert.net/)
- [RAG Paper Original](https://arxiv.org/abs/2005.11401)

---

**¡Listo para tu masterclass!** 🚀

Este documento cubre todos los conceptos fundamentales que necesitas explicar. Puedes usarlo como guía durante la presentación o compartirlo con los participantes.

