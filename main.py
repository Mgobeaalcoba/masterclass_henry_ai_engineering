#!/usr/bin/env python
"""
Chat con RAG - Demostración de LLM con contexto
Usa LangChain + ChromaDB para dar contexto al modelo sobre documentación privada
"""

import os
from dotenv import load_dotenv

# Imports de LangChain para RAG
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Cargar variables de entorno
load_dotenv()


def configurar_rag():
    """
    Configura el sistema RAG: carga documentos, crea vectores y configura la cadena QA
    """
    print("\n🔧 Configurando RAG (Retrieval Augmented Generation)...\n")
    
    # 1. Cargar el documento de documentación técnica
    print("📄 Cargando documento: documentacion_tecnica.md")
    loader = TextLoader("documentacion_tecnica.md", encoding="utf-8")
    documents = loader.load()
    print(f"   ✅ Documento cargado ({len(documents)} archivo(s))\n")
    
    # 2. Dividir el documento en chunks para mejor recuperación
    print("✂️  Dividiendo documento en fragmentos...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   ✅ {len(chunks)} fragmentos creados\n")
    
    # 3. Crear embeddings y almacenar en ChromaDB
    print("🧠 Creando embeddings (HuggingFace - 100% GRATIS) y base vectorial...")
    print("   ⏳ Primera vez puede tomar un momento (descarga modelo ~400MB)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("   ✅ Base vectorial creada (sin costo!)\n")
    
    # 4. Detectar modelo disponible
    modelos = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
    modelo_actual = None
    
    print("🔍 Detectando modelo disponible...")
    for modelo in modelos:
        try:
            test_llm = ChatOpenAI(model=modelo, temperature=0, max_tokens=5)
            test_llm.invoke("test")
            modelo_actual = modelo
            print(f"   ✅ Usando modelo: {modelo}\n")
            break
        except:
            continue
    
    if not modelo_actual:
        raise Exception("No se encontró ningún modelo disponible")
    
    # 5. Configurar el LLM
    llm = ChatOpenAI(
        model=modelo_actual,
        temperature=0.7,
        max_tokens=500
    )
    
    # 6. Crear retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 7. Crear prompt template
    template = """Eres un asistente útil que responde preguntas basándote en la documentación proporcionada.

Contexto de la documentación:
{context}

Pregunta del usuario: {question}

Responde la pregunta basándote ÚNICAMENTE en el contexto proporcionado. Si la información no está en el contexto, di que no tienes esa información."""

    prompt = ChatPromptTemplate.from_template(template)
    
    # 8. Función para formatear documentos
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # 9. Crear la cadena RAG
    print("🔗 Configurando cadena de pregunta-respuesta...")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("   ✅ Sistema RAG listo\n")
    
    return rag_chain, modelo_actual, retriever


def main():
    """
    Función principal: configura RAG y ejecuta el chat interactivo
    """
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: Configura OPENAI_API_KEY en el archivo .env")
        return
    
    try:
        rag_chain, modelo_actual, retriever = configurar_rag()
        
        print("="*70)
        print("  💬 Chat con RAG - El modelo AHORA tiene contexto")
        print("="*70)
        print("\n💡 El modelo tiene acceso a documentacion_tecnica.md")
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
                print(f"\n⏳ Buscando en documentación y consultando {modelo_actual}...\n")
                
                # Ejecutar la consulta con RAG
                respuesta = rag_chain.invoke(pregunta)
                
                print(f"🤖 {modelo_actual.upper()} (CON CONTEXTO): {respuesta}\n")
                
                # Mostrar fragmentos recuperados (opcional)
                docs = retriever.invoke(pregunta)
                print(f"📚 Fuentes: {len(docs)} fragmentos consultados")
                
                print("-" * 70 + "\n")
                
            except Exception as e:
                print(f"❌ Error: {e}\n")
    
    except Exception as e:
        print(f"❌ Error al configurar RAG: {e}\n")
        print("💡 Asegúrate de que:")
        print("   - El archivo documentacion_tecnica.md existe")
        print("   - Tu API Key de OpenAI es válida")
        print("   - Tienes las dependencias instaladas (poetry install)\n")


if __name__ == "__main__":
    main()
