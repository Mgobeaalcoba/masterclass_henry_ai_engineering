#!/usr/bin/env python
"""
Demostración: LLM HÍBRIDO - RAG + Conocimiento del Modelo

Combina RAG con conocimiento del entrenamiento del modelo:
1. Si está en la documentación → Responde con RAG
2. Si NO está en la documentación → Responde con conocimiento del modelo
3. Si tampoco sabe → Dice "No sé"
"""

import os
# Configurar tokenizers para evitar warnings de paralelismo después de fork
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Configuración
DOCUMENTO = "documentacion_tecnica.md"
CHROMA_DB_DIR = "./chroma_db"

def detectar_modelo():
    """Detecta qué modelo de OpenAI está disponible en tu cuenta."""
    for modelo in ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]:
        try:
            ChatOpenAI(model=modelo, temperature=0, max_tokens=5).invoke("test")
            return modelo
        except:
            continue
    return None

def configurar_sistema_hibrido():
    """
    Configura el sistema híbrido: RAG + conocimiento del modelo.
    
    Returns:
        tuple: (retriever, llm, modelo_actual, vectorstore)
    """
    print("\n🔧 Configurando Sistema Híbrido (RAG + Conocimiento del Modelo)...\n")
    
    # 1. Cargar documento
    print("📄 Cargando documento:", DOCUMENTO)
    documentos = TextLoader(DOCUMENTO, encoding="utf-8").load()
    print(f"   ✅ Documento cargado ({len(documentos)} archivo(s))\n")
    
    # 2. Dividir en fragmentos
    print("✂️  Dividiendo documento en fragmentos...")
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len).split_documents(documentos)
    print(f"   ✅ {len(chunks)} fragmentos creados\n")
    
    # 3. Crear embeddings y base vectorial
    print("🧠 Creando embeddings (HuggingFace - 100% GRATIS) y base vectorial...")
    print("   ⏳ Primera vez puede tomar un momento (descarga modelo ~400MB)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_DB_DIR)
    print("   ✅ Base vectorial creada (sin costo!)\n")
    
    # 4. Detectar y configurar modelo
    print("🔍 Detectando modelo disponible...")
    modelo = detectar_modelo()
    if not modelo:
        raise Exception("No se encontró ningún modelo disponible")
    print(f"   ✅ Usando modelo: {modelo}\n")
    
    # Configuración anti-alucinación
    llm = ChatOpenAI(model=modelo, temperature=0.1, max_tokens=500)
    
    # 5. Crear retriever (busca los 3 fragmentos más relevantes)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    print("   ✅ Sistema híbrido listo\n")
    
    return retriever, llm, modelo, vectorstore

def evaluar_relevancia_documentos(docs, pregunta):
    """
    Evalúa si los documentos recuperados son relevantes para la pregunta.
    
    Estrategia mejorada: Verifica si hay coincidencias significativas de palabras clave
    específicas del tema. Si encuentra documentos con contenido sustancial, los considera relevantes.
    
    Args:
        docs: Lista de documentos recuperados
        pregunta: Pregunta original del usuario
        
    Returns:
        bool: True si hay información relevante, False si no
    """
    if not docs or len(docs) == 0:
        return False
    
    # Verificar contenido de documentos
    contenido_texto = " ".join(doc.page_content.lower() for doc in docs)
    contenido_total = len(contenido_texto)
    
    # Si el contenido total es muy pequeño, no es relevante
    if contenido_total < 50:
        return False
    
    # Extraer palabras clave de la pregunta (palabras significativas)
    pregunta_lower = pregunta.lower()
    palabras_pregunta = set(pregunta_lower.split())
    
    # Palabras comunes a ignorar (stopwords en español)
    palabras_ignorar = {
        'qué', 'cómo', 'cuándo', 'dónde', 'por', 'para', 'con', 'de', 'la', 'el', 'un', 'una',
        'es', 'son', 'está', 'están', 'sobre', 'el', 'la', 'los', 'las', 'un', 'una',
        'y', 'o', 'pero', 'si', 'no', 'en', 'a', 'de', 'que', 'se', 'le', 'te', 'me', 'nos', 'les',
        'puedes', 'puede', 'puedo', 'pueden', 'pueda', 'puedan', 'para', 'que', 'sirve', 'sirven',
        'consultar', 'fuera', 'fuentes', 'tus', 'sus', 'mis', 'nuestros', 'vuestros',
        'trata', 'se', 'trata', 'como', 'instala', 'instalar'
    }
    
    palabras_clave = palabras_pregunta - palabras_ignorar
    
    # Si hay contenido sustancial (más de 200 caracteres), considerar relevante por defecto
    # El retriever ya hizo el trabajo de encontrar documentos similares
    if contenido_total > 200:
        # Verificar si hay al menos una palabra clave importante en el contenido
        # Esto evita casos donde el contenido es grande pero no relacionado
        palabras_importantes = [p for p in palabras_clave if len(p) > 3]  # Palabras de más de 3 caracteres
        if len(palabras_importantes) == 0:
            # Si no hay palabras importantes, pero el contenido es sustancial, confiar en el retriever
            return True
        
        # Verificar si alguna palabra importante aparece
        tiene_coincidencias = any(palabra in contenido_texto for palabra in palabras_importantes)
        if tiene_coincidencias:
            return True
    
    # Si el contenido es menor pero hay coincidencias significativas
    if contenido_total >= 50:
        coincidencias = sum(1 for palabra in palabras_clave if len(palabra) > 2 and palabra in contenido_texto)
        if coincidencias > 0:
            return True
    
    return False

def responder_con_rag(pregunta, retriever, llm):
    """
    Responde usando RAG cuando hay información en la documentación.
    
    Args:
        pregunta: Pregunta del usuario
        retriever: Retriever de documentos
        llm: Modelo de lenguaje
        
    Returns:
        str: Respuesta generada con RAG
    """
    template_rag = """Eres un asistente útil que responde preguntas basándote ÚNICAMENTE en la documentación proporcionada.

Contexto de la documentación:
{context}

Pregunta del usuario: {question}

INSTRUCCIONES CRÍTICAS:
- DEBES responder usando la información del contexto proporcionado
- Si el contexto contiene información sobre el tema, úsala para responder de manera completa
- Extrae TODA la información relevante del contexto y preséntala de forma clara
- NO uses conocimiento fuera del contexto proporcionado
- Si el contexto NO contiene información sobre la pregunta, entonces di "No tengo información sobre esto en la documentación"
- Sé exhaustivo: si hay información en el contexto, úsala toda"""
    
    prompt_rag = ChatPromptTemplate.from_template(template_rag)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_rag
        | llm
        | StrOutputParser()
    )
    
    return rag_chain.invoke(pregunta)

def responder_con_rag_directo(pregunta, docs, llm):
    """
    Responde usando RAG con documentos ya recuperados (fallback cuando el prompt normal falla).
    
    Args:
        pregunta: Pregunta del usuario
        docs: Documentos ya recuperados
        llm: Modelo de lenguaje
        
    Returns:
        str: Respuesta generada con RAG
    """
    template_directo = """Responde la siguiente pregunta usando la información proporcionada en la documentación.

Documentación:
{context}

Pregunta: {question}

Responde usando SOLO la información de la documentación. Si hay información relevante, úsala para responder completamente."""
    
    prompt_directo = ChatPromptTemplate.from_template(template_directo)
    
    contexto = "\n\n".join(doc.page_content for doc in docs)
    
    # Crear mensaje directamente
    mensaje = prompt_directo.format(context=contexto, question=pregunta)
    respuesta = llm.invoke(mensaje)
    
    return respuesta.content if hasattr(respuesta, 'content') else str(respuesta)

def responder_con_conocimiento_propio(pregunta, llm):
    """
    Responde usando el conocimiento del entrenamiento del modelo.
    
    Args:
        pregunta: Pregunta del usuario
        llm: Modelo de lenguaje
        
    Returns:
        str: Respuesta generada con conocimiento del modelo
    """
    template_propio = """Eres un asistente útil y honesto. Responde la pregunta usando tu conocimiento de entrenamiento.

Pregunta: {question}

INSTRUCCIONES:
- Responde usando tu conocimiento general si lo tienes
- Si NO sabes la respuesta, di claramente "No sé sobre..." o "No tengo información sobre..."
- NO inventes información. Sé honesto y directo."""
    
    prompt_propio = ChatPromptTemplate.from_template(template_propio)
    
    chain_propio = (
        {"question": RunnablePassthrough()}
        | prompt_propio
        | llm
        | StrOutputParser()
    )
    
    return chain_propio.invoke(pregunta)

def responder_hibrido(pregunta, retriever, llm):
    """
    Responde usando estrategia híbrida: primero RAG, luego conocimiento propio.
    
    Args:
        pregunta: Pregunta del usuario
        retriever: Retriever de documentos
        llm: Modelo de lenguaje
        
    Returns:
        tuple: (respuesta, fuente_usada)
    """
    # Detectar si el usuario explícitamente pide usar conocimiento fuera de las fuentes
    pregunta_lower = pregunta.lower()
    palabras_fuera = {'fuera', 'fuentes', 'consultar', 'por fuera', 'sin documentación', 'conocimiento propio', 'entrenamiento'}
    pedir_fuera = any(palabra in pregunta_lower for palabra in palabras_fuera)
    
    # Si el usuario explícitamente pide usar conocimiento fuera, hacerlo directamente
    if pedir_fuera:
        # Limpiar la pregunta removiendo las palabras de "fuera"
        pregunta_limpia = pregunta
        for palabra in palabras_fuera:
            pregunta_limpia = pregunta_limpia.replace(palabra, "").strip()
        respuesta = responder_con_conocimiento_propio(pregunta_limpia if pregunta_limpia else pregunta, llm)
        return respuesta, "conocimiento del modelo"
    
    # 1. Buscar en documentación
    docs = retriever.invoke(pregunta)
    
    # 2. Evaluar si hay información relevante
    if evaluar_relevancia_documentos(docs, pregunta):
        # Usar RAG con documentación - confiar en los documentos encontrados
        respuesta = responder_con_rag(pregunta, retriever, llm)
        
        # Si encontramos documentos relevantes, confiar en RAG
        # Solo cambiar a conocimiento propio si la respuesta es muy corta o claramente indica falta de info
        respuesta_lower = respuesta.lower()
        tiene_info_insuficiente = (
            ("no tengo información" in respuesta_lower or "no sé" in respuesta_lower) 
            and len(respuesta) < 50  # Respuesta muy corta
        )
        
        # Si RAG dice que no tiene info pero encontramos documentos relevantes, 
        # es probable que el prompt no esté funcionando bien, pero aún así confiar en RAG
        # porque los documentos SÍ tienen información
        if tiene_info_insuficiente:
            # Intentar una vez más con un prompt más directo
            respuesta_directa = responder_con_rag_directo(pregunta, docs, llm)
            if len(respuesta_directa) > 50:  # Si la respuesta directa tiene contenido
                return respuesta_directa, "documentación"
        
        return respuesta, "documentación"
    else:
        # Usar conocimiento propio del modelo
        respuesta = responder_con_conocimiento_propio(pregunta, llm)
        return respuesta, "conocimiento del modelo"

def main():
    """
    Función principal: Sistema híbrido que combina RAG con conocimiento del modelo.
    
    Estrategia:
    1. Busca primero en la documentación (RAG)
    2. Si no encuentra información relevante, usa conocimiento del modelo
    3. Si tampoco sabe, el modelo dice "No sé"
    """
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: Configura OPENAI_API_KEY en el archivo .env")
        return
    
    try:
        retriever, llm, modelo, vectorstore = configurar_sistema_hibrido()
        
        print("="*70)
        print("  💬 Chat HÍBRIDO - RAG + Conocimiento del Modelo")
        print("="*70)
        print(f"\n💡 El modelo busca primero en {DOCUMENTO}")
        print("💡 Si no encuentra información, usa su conocimiento de entrenamiento")
        print("💡 Escribe 'salir' para terminar\n")
        print("🎯 Prueba preguntando:")
        print("   - ¿Cómo instalo la librería HenryPy? (está en documentación)")
        print("   - ¿Qué es Python? (conocimiento general)\n")
        
        while True:
            pregunta = input("🧑 TÚ: ").strip()
            
            if pregunta.lower() in ['salir', 'exit', 'quit']:
                print("\n👋 ¡Hasta luego!\n")
                break
            
            if not pregunta:
                continue
            
            try:
                print(f"\n⏳ Analizando pregunta y consultando {modelo}...\n")
                
                # Responder con estrategia híbrida
                respuesta, fuente = responder_hibrido(pregunta, retriever, llm)
                
                # Mostrar respuesta con indicador de fuente
                fuente_emoji = "📚" if fuente == "documentación" else "🧠"
                fuente_texto = "DOCUMENTACIÓN" if fuente == "documentación" else "CONOCIMIENTO PROPIO"
                
                print(f"🤖 {modelo.upper()} ({fuente_emoji} {fuente_texto}): {respuesta}\n")
                
                # Mostrar documentos consultados si usó RAG
                if fuente == "documentación":
                    docs = retriever.invoke(pregunta)
                    print(f"📚 Fuentes: {len(docs)} fragmentos consultados de la documentación")
                
                print("-" * 70 + "\n")
                
            except Exception as e:
                print(f"❌ Error: {e}\n")
    
    except Exception as e:
        print(f"❌ Error al configurar sistema híbrido: {e}\n")
        print("💡 Asegúrate de que:")
        print(f"   - El archivo {DOCUMENTO} existe")
        print("   - Tu API Key de OpenAI es válida")
        print("   - Tienes las dependencias instaladas (poetry install)\n")

if __name__ == "__main__":
    main()

