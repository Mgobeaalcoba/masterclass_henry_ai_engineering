# 🧠 Masterclass: Cerebro Corporativo con RAG

> **Demostración práctica de cómo darle "memoria" a los LLMs usando RAG (Retrieval Augmented Generation)**

## 📖 Índice

- [¿Qué es este proyecto?](#-qué-es-este-proyecto)
- [El Problema que Resuelve](#-el-problema-que-resuelve)
- [La Solución: RAG](#-la-solución-rag)
- [Instalación](#-instalación)
- [Configuración](#-configuración)
- [Cómo Usar](#-cómo-usar)
- [Estructura de la Masterclass](#-estructura-de-la-masterclass)
- [Arquitectura Técnica](#-arquitectura-técnica)
- [Costos](#-costos)
- [Tips para Presentadores](#-tips-para-presentadores)

---

## 🎯 ¿Qué es este proyecto?

Este proyecto es una **demostración didáctica** diseñada para enseñar cómo implementar un **Cerebro Corporativo** utilizando RAG (Retrieval Augmented Generation). 

En **30 minutos**, los participantes aprenderán:
- ✅ Por qué los LLMs no conocen datos privados de empresas
- ✅ Cómo funciona RAG (Retrieval Augmented Generation)
- ✅ Implementación práctica con LangChain + ChromaDB
- ✅ La diferencia entre un modelo CON y SIN contexto

### 🎓 Contexto Educativo

Este proyecto está diseñado para una **Masterclass de Henry** sobre cómo construir sistemas de IA que pueden acceder y utilizar documentación interna de una empresa.

---

## ❌ El Problema que Resuelve

### Situación Real en Empresas

Los modelos de lenguaje más potenttes del mundo (GPT-4, Claude, etc.) **NO conocen**:
- 📋 Documentación interna de tu empresa
- 🔐 Políticas y procedimientos privados
- 📚 Manuales de librerías propietarias
- 💼 Información de productos internos

### Ejemplo Práctico

Pregunta: **"¿Cómo instalo la librería HenryPy?"**

```
🤖 GPT-4 SIN contexto:
"Lo siento, no tengo información sobre una librería llamada 'HenryPy'..."

🤖 GPT-4 CON RAG:
"Para instalar HenryPy, usa: pip install henrypy
Para el motor de análisis avanzado: pip install henrypy[analysis]"
```

---

## ✅ La Solución: RAG

**RAG (Retrieval Augmented Generation)** es una técnica que combina:

```
┌─────────────────────────────────────────────────────────────┐
│                    FLUJO DE RAG                             │
└─────────────────────────────────────────────────────────────┘

1. INDEXACIÓN (Una sola vez)
   📄 Documentos Privados
   ↓
   ✂️  División en fragmentos
   ↓
   🧠 Creación de embeddings (vectores)
   ↓
   💾 Almacenamiento en ChromaDB

2. CONSULTA (Cada pregunta)
   🧑 Usuario: "¿Cómo instalo HenryPy?"
   ↓
   🔍 Búsqueda semántica en ChromaDB
   ↓
   📚 Recupera fragmentos relevantes
   ↓
   📝 Construye prompt: [Contexto + Pregunta]
   ↓
   🤖 LLM genera respuesta precisa
   ↓
   ✅ "Para instalar HenryPy, usa pip install henrypy"
```

---

## 🚀 Instalación

### Requisitos Previos

- **Python 3.13+**
- **Poetry** (gestor de dependencias)
- **API Key de OpenAI** (para generar respuestas)

### 1. Clonar o Descargar el Proyecto

```bash
cd /tu/directorio
# (el proyecto ya debería estar aquí)
```

### 2. Instalar Poetry (si no lo tienes)

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 3. Instalar Dependencias

```bash
cd masterclass_henry_cerebro_corpo
poetry install
```

Esto instalará:
- ✅ LangChain (framework para LLMs)
- ✅ ChromaDB (base de datos vectorial)
- ✅ HuggingFace (embeddings gratuitos)
- ✅ OpenAI (para generar respuestas)
- ✅ Sentence-Transformers (modelos de embeddings)

---

## ⚙️ Configuración

### Crear archivo `.env`

Crea un archivo `.env` en la raíz del proyecto:

```bash
nano .env
```

Agrega tu API Key de OpenAI:

```env
OPENAI_API_KEY=sk-tu-clave-aqui
```

**¿Dónde obtener la API Key?**
- Ve a: https://platform.openai.com/api-keys
- Crea una nueva clave
- Cópiala al archivo `.env`

**💡 Tip:** Para la demo, necesitas al menos **$5 USD** en créditos de OpenAI.

---

## 💻 Cómo Usar

### 📂 Archivos del Proyecto

```
masterclass_henry_cerebro_corpo/
├── model_without_context.py    # 1️⃣ Demostración SIN contexto
├── main.py                      # 2️⃣ Demostración CON contexto (RAG)
├── documentacion_tecnica.md     # 📚 Documentación de HenryPy (datos privados)
├── .env                         # 🔑 API Key de OpenAI
└── README.md                    # 📖 Este archivo
```

### 1️⃣ Demostración SIN Contexto

**Archivo:** `model_without_context.py`

Muestra cómo GPT-4 **NO puede responder** sobre datos privados:

```bash
poetry run python model_without_context.py
```

**Pregunta sugerida:**
```
¿Cómo instalo la librería HenryPy?
```

**Resultado esperado:**
```
🤖 GPT-4O-MINI: No tengo información sobre una librería llamada "HenryPy"...
```

### 2️⃣ Demostración CON Contexto (RAG)

**Archivo:** `main.py`

Muestra cómo GPT-4 **SÍ puede responder** usando RAG:

```bash
poetry run python main.py
```

**Primera ejecución:**
- ⏳ Descarga modelo de embeddings (~400MB) - toma 2-5 min
- 📄 Carga `documentacion_tecnica.md`
- ✂️ Divide en fragmentos
- 🧠 Crea embeddings (100% GRATIS con HuggingFace)
- 💾 Crea base vectorial en ChromaDB

**La MISMA pregunta:**
```
¿Cómo instalo la librería HenryPy?
```

**Resultado esperado:**
```
🤖 GPT-4O-MINI (CON CONTEXTO): Para instalar HenryPy, 
ejecuta: pip install henrypy

Si necesitas el motor de análisis avanzado:
pip install henrypy[analysis]
```

### 🎯 Más Preguntas para Probar

Todas estas funcionan CON contexto, pero NO sin contexto:

```
¿Qué es HenryPy?
¿Cómo configuro la API Key de HenryPy?
¿Qué hace la función henrypy.analyze()?
¿Qué error da si no instalo henrypy[analysis]?
¿HenryPy envía mi código a servidores externos?
```

---

## 🎓 Estructura de la Masterclass

### Formato: 30 Minutos

#### **Fase 1: El Problema (5 min)**

1. **Demostrar la limitación** (2 min)
   ```bash
   poetry run python model_without_context.py
   ```
   - Pregunta: "¿Cómo instalo HenryPy?"
   - GPT no sabe responder

2. **Explicar el problema** (3 min)
   - Los LLMs solo conocen datos públicos
   - No tienen acceso a información privada
   - Esto es un problema en empresas

#### **Fase 2: La Solución (10 min)**

3. **Introducir RAG** (3 min)
   - ¿Qué es RAG?
   - Los 3 componentes: Documentos → Vectores → Búsqueda
   - Diagrama en pizarra

4. **Demostrar RAG funcionando** (7 min)
   ```bash
   poetry run python main.py
   ```
   - Mostrar proceso de indexación
   - Hacer la misma pregunta
   - Explicar qué pasó detrás de escena

#### **Fase 3: Implementación (15 min)**

5. **Revisar el código** (10 min)
   - Abrir `main.py` en editor
   - Explicar cada componente:
     - Document Loader
     - Text Splitter
     - Embeddings (HuggingFace)
     - ChromaDB
     - RAG Chain (LCEL)

6. **Q&A y Casos de Uso** (5 min)
   - Preguntas de la audiencia
   - Ejemplos de uso en empresas reales

---

## 🏗️ Arquitectura Técnica

### Stack Tecnológico

```
┌─────────────────────────────────────┐
│     🧑 Usuario (Terminal)           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│     📝 LangChain (Orquestación)     │
│  • Manejo del flujo RAG             │
│  • Construcción de prompts          │
│  • Integración de componentes       │
└──────────────┬──────────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌──────────────┐  ┌──────────────────┐
│ 🧠 Embeddings │  │  🤖 OpenAI GPT   │
│  HuggingFace  │  │  • gpt-4o-mini   │
│  • GRATIS     │  │  • Generación    │
│  • Local      │  │  • Respuestas    │
└──────┬───────┘  └──────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│     💾 ChromaDB (Vector Store)       │
│  • Almacena embeddings               │
│  • Búsqueda por similitud            │
│  • Persiste en disco                 │
└──────────────────────────────────────┘
```

### Componentes Clave

| Componente | Función | Costo |
|------------|---------|-------|
| **LangChain** | Orquestación del flujo RAG | 🆓 Gratis |
| **HuggingFace** | Crear embeddings (vectores) | 🆓 Gratis |
| **ChromaDB** | Almacenar y buscar vectores | 🆓 Gratis |
| **OpenAI GPT** | Generar respuestas finales | 💰 Requiere créditos |

---

## 💰 Costos

### Breakdown de Costos

#### 🆓 Componentes GRATUITOS

- ✅ **Embeddings**: HuggingFace (100% gratis, corre local)
- ✅ **Vector Store**: ChromaDB (código abierto)
- ✅ **Framework**: LangChain (código abierto)

#### 💰 Componentes de PAGO

- **OpenAI API** (solo para generar respuestas):
  - Crear embeddings del documento: ~$0.01
  - 50 preguntas de prueba: ~$0.10
  - **Total para la demo: < $1 USD**

### Para la Masterclass

**Opción 1: Con $5 USD**
- ✅ Demo completa funcionando
- ✅ Puedes responder preguntas en vivo
- ✅ Muestra el sistema end-to-end

**Opción 2: Sin créditos**
- ✅ Muestra el código
- ✅ Explica conceptualmente
- ⚠️ No ejecuta en vivo
- 💡 Usa screenshots preparados

---

## 🎤 Tips para Presentadores

### Preparación (1 día antes)

1. **✅ Probar ambos scripts**
   ```bash
   poetry run python model_without_context.py
   poetry run python main.py
   ```

2. **✅ Primera ejecución de `main.py`**
   - Descarga el modelo (toma tiempo)
   - Crea la base vectorial
   - Prueba varias preguntas

3. **✅ Preparar screenshots de backup**
   - Por si falla la conexión
   - Por si hay problemas técnicos

### Durante la Presentación

#### DO's ✅

- ✅ **Empezar con el problema** (model_without_context.py)
- ✅ **Hacer la misma pregunta dos veces** (sin y con contexto)
- ✅ **Mostrar el código** (main.py abierto en editor)
- ✅ **Explicar cada componente** mientras señalas el código
- ✅ **Permitir 2-3 preguntas del público** durante la demo

#### DON'Ts ❌

- ❌ No te enfoques en detalles técnicos complejos
- ❌ No uses jerga sin explicar
- ❌ No te saltes la demo del problema
- ❌ No asumas que todos saben qué son los embeddings

### Si Algo Falla

#### Error: "insufficient_quota"
> "Perfecto, esto me permite mostrarles algo sobre arquitectura: 
> en producción, implementarían rate limiting, caching, y fallbacks.
> Por ahora, veamos el código y explico el flujo..."

#### Error: Conexión lenta
> "Mientras carga, déjenme explicarles qué está pasando detrás de escena:
> está creando vectores de cada fragmento del documento..."

#### Sin conexión
> "Tengo screenshots preparados de la ejecución. Lo importante es 
> entender el concepto..."

---

## 📚 Recursos Adicionales

### Documentación del Proyecto

- **`EMBEDDINGS_GRATUITOS.md`** - Cómo usar embeddings sin costo
- **`CAMBIOS_LANGCHAIN.md`** - Explicación de la API moderna de LangChain
- **`EXPLICACION_TECNICA_RAG.md`** - Deep dive técnico de RAG

### Para Estudiantes

Después de la masterclass, los estudiantes pueden:
1. 📖 Leer la documentación técnica incluida
2. 🔧 Modificar el código para usar sus propios documentos
3. 🚀 Experimentar con diferentes preguntas
4. 🎯 Implementar RAG en sus propios proyectos

### Recursos Externos

- **LangChain Docs**: https://python.langchain.com/
- **ChromaDB Docs**: https://docs.trychroma.com/
- **Sentence Transformers**: https://www.sbert.net/
- **OpenAI API**: https://platform.openai.com/docs

---

## 🤝 Contribuciones y Feedback

Este es un proyecto educativo. Si tienes sugerencias para mejorar la masterclass:

1. 💬 Comparte tu feedback
2. 🐛 Reporta problemas técnicos
3. 💡 Sugiere mejoras pedagógicas
4. 📝 Comparte casos de uso reales

---

## 📄 Licencia

Este proyecto está diseñado con fines educativos para la comunidad de Henry.

---

## 🎓 Sobre la Masterclass

**Objetivo:** Enseñar cómo construir un "Cerebro Corporativo" que permite a los LLMs acceder y utilizar documentación interna de empresas.

**Duración:** 30 minutos

**Nivel:** Intermedio (conocimientos básicos de Python y LLMs)

**Resultado:** Los participantes salen sabiendo cómo implementar RAG en sus propios proyectos.

---

## 🚀 ¡Comienza Ahora!

```bash
# 1. Instalar dependencias
poetry install

# 2. Configurar API Key
echo "OPENAI_API_KEY=tu-clave-aqui" > .env

# 3. Probar sin contexto
poetry run python model_without_context.py

# 4. Probar con RAG
poetry run python main.py

# 5. ¡Listo para la masterclass! 🎉
```

---

**¿Preguntas?** Revisa la documentación técnica en los archivos `.md` incluidos.

**¡Buena suerte con tu masterclass!** 🎓✨
