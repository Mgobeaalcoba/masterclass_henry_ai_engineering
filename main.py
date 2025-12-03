#!/usr/bin/env python
"""
Demostración: LLM CON contexto usando RAG (Retrieval Augmented Generation)

Muestra cómo darle contexto a GPT sobre documentación privada usando LangChain + ChromaDB.
El modelo ahora puede responder sobre datos internos que no están en su entrenamiento.
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

def configurar_rag():
    """
    Configura el sistema RAG completo.
    
    Pasos: Carga documento → Divide en chunks → Crea embeddings → Almacena en ChromaDB → 
    Configura cadena de pregunta-respuesta con contexto.
    
    Returns:
        tuple: (rag_chain, modelo_actual, retriever)
    """
    print("\n🔧 Configurando RAG (Retrieval Augmented Generation)...\n")
    
    # 1. Cargar documento
    print("📄 Cargando documento:", DOCUMENTO)
    documentos = TextLoader(DOCUMENTO, encoding="utf-8").load()
    print(f"   ✅ Documento cargado ({len(documentos)} archivo(s))\n")
    
    # 2. Dividir en fragmentos para mejor recuperación
    print("✂️  Dividiendo documento en fragmentos...")
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len).split_documents(documentos)
    print(f"   ✅ {len(chunks)} fragmentos creados\n")
    
    # 3. Crear embeddings y base vectorial (HuggingFace es gratis)
    print("🧠 Creando embeddings (HuggingFace - 100% GRATIS) y base vectorial...")
    print("   ⏳ Primera vez puede tomar un momento (descarga modelo ~400MB)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_DB_DIR)
    print("   ✅ Base vectorial creada (sin costo!)\n")
    
    # 4. Detectar y configurar modelo de chat
    print("🔍 Detectando modelo disponible...")
    modelo = detectar_modelo()
    if not modelo:
        raise Exception("No se encontró ningún modelo disponible")
    print(f"   ✅ Usando modelo: {modelo}\n")
    # Configuración anti-alucinación: temperatura muy baja para reducir creatividad y alucinaciones
    llm = ChatOpenAI(model=modelo, temperature=0.1, max_tokens=500)
    
    # 5. Crear retriever (busca los 3 fragmentos más relevantes)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 6. Crear prompt template para el contexto (más estricto para evitar alucinaciones)
    template = """Eres un asistente útil y preciso que responde preguntas basándote ÚNICAMENTE en la documentación proporcionada.

Contexto de la documentación:
{context}

Pregunta del usuario: {question}

INSTRUCCIONES IMPORTANTES:
- Responde SOLO usando la información que está en el contexto proporcionado
- Si la información NO está en el contexto, di claramente "No tengo información sobre esto en la documentación" o "No sé sobre..."
- NO inventes, NO supongas, NO uses conocimiento general fuera del contexto
- Sé preciso y directo. Si no sabes algo, dilo claramente."""
    prompt = ChatPromptTemplate.from_template(template)
    
    # 7. Función para formatear documentos recuperados
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # 8. Crear cadena RAG (LangChain Expression Language)
    print("🔗 Configurando cadena de pregunta-respuesta...")
    rag_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    print("   ✅ Sistema RAG listo\n")
    
    return rag_chain, modelo, retriever

def main():
    """
    Función principal: Configura RAG y ejecuta chat interactivo con contexto.
    
    Demuestra cómo el modelo ahora SÍ puede responder sobre datos privados
    porque tiene acceso a la documentación técnica mediante RAG.
    """
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: Configura OPENAI_API_KEY en el archivo .env")
        return
    
    try:
        rag_chain, modelo, retriever = configurar_rag()
        
        print("="*70)
        print("  💬 Chat con RAG - El modelo AHORA tiene contexto")
        print("="*70)
        print(f"\n💡 El modelo tiene acceso a {DOCUMENTO}")
        print("💡 Escribe 'salir' para terminar\n")
        print("🎯 Prueba preguntando: ¿Cómo instalo la librería HenryPy?\n")
        
        while True:
            pregunta = input("🧑 TÚ: ").strip()
            
            if pregunta.lower() in ['salir', 'exit', 'quit']:
                print("\n👋 ¡Hasta luego!\n")
                break
            
            if not pregunta:
                continue
            
            try:
                print(f"\n⏳ Buscando en documentación y consultando {modelo}...\n")
                respuesta = rag_chain.invoke(pregunta)
                print(f"🤖 {modelo.upper()} (CON CONTEXTO): {respuesta}\n")
                print(f"📚 Fuentes: {len(retriever.invoke(pregunta))} fragmentos consultados")
                print("-" * 70 + "\n")
            except Exception as e:
                print(f"❌ Error: {e}\n")
    
    except Exception as e:
        print(f"❌ Error al configurar RAG: {e}\n")
        print("💡 Asegúrate de que:")
        print(f"   - El archivo {DOCUMENTO} existe")
        print("   - Tu API Key de OpenAI es válida")
        print("   - Tienes las dependencias instaladas (poetry install)\n")

if __name__ == "__main__":
    main()
