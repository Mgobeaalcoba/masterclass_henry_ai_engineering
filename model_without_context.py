#!/usr/bin/env python
"""
Chat simple con GPT-4 para demostrar falta de contexto
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

def main():
    # Verificar API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: Configura OPENAI_API_KEY en el archivo .env")
        return
    
    # Inicializar cliente
    client = OpenAI()
    
    # Lista de modelos para probar (del más avanzado al más accesible)
    modelos = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
    modelo_actual = None
    
    # Detectar qué modelo está disponible
    print("\n🔍 Detectando modelo disponible...")
    for modelo in modelos:
        try:
            test = client.chat.completions.create(
                model=modelo,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            modelo_actual = modelo
            print(f"✅ Usando modelo: {modelo}\n")
            break
        except:
            continue
    
    if not modelo_actual:
        print("❌ No se encontró ningún modelo disponible en tu cuenta")
        print("💡 Verifica tu API Key o actualiza tu plan en OpenAI")
        return
    
    print("="*60)
    print("  💬 Chat con OpenAI - Demostración de Falta de Contexto")
    print("="*60)
    print("\n💡 Escribe 'salir' para terminar\n")
    
    # Loop de conversación
    while True:
        # Obtener pregunta del usuario
        pregunta = input("🧑 TÚ: ").strip()
        
        if pregunta.lower() in ['salir', 'exit', 'quit']:
            print("\n👋 ¡Hasta luego!\n")
            break
            
        if not pregunta:
            continue
        
        # Consultar al modelo
        try:
            print(f"\n⏳ Consultando {modelo_actual}...\n")
            
            response = client.chat.completions.create(
                model=modelo_actual,
                messages=[{"role": "user", "content": pregunta}],
                temperature=0.7,
                max_tokens=500
            )
            
            respuesta = response.choices[0].message.content
            print(f"🤖 {modelo_actual.upper()}: {respuesta}\n")
            print("-" * 60 + "\n")
            
        except Exception as e:
            print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    main()
