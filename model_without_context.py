#!/usr/bin/env python
"""
Demostración: LLM sin contexto de datos privados

Muestra cómo GPT-4 NO puede responder sobre información privada de empresas
porque no tiene acceso a documentación interna.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Cargar API Key desde archivo .env

def detectar_modelo(client):
    """Detecta qué modelo de OpenAI está disponible en tu cuenta."""
    for modelo in ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]:
        try:
            client.chat.completions.create(model=modelo, messages=[{"role": "user", "content": "test"}], max_tokens=5)
            return modelo
        except:
            continue
    return None

def main():
    """Chat interactivo con GPT sin contexto - Demuestra la falta de conocimiento sobre datos privados."""
    
    # Verificar API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: Configura OPENAI_API_KEY en el archivo .env")
        return
    
    # Inicializar cliente y detectar modelo
    client = OpenAI()
    print("\n🔍 Detectando modelo disponible...")
    modelo = detectar_modelo(client)
    
    if not modelo:
        print("❌ No se encontró ningún modelo disponible")
        return
    
    print(f"✅ Usando modelo: {modelo}\n")
    print("="*60)
    print("  💬 Chat SIN Contexto - El modelo NO conoce datos privados")
    print("="*60)
    print("\n💡 Escribe 'salir' para terminar\n")
    
    # Prompt del sistema para evitar alucinaciones
    system_prompt = """Eres un asistente honesto y directo. Si no conoces o no tienes información sobre algo, 
debes decir claramente "No sé sobre..." o "No tengo información sobre..." en lugar de inventar o suponer.
Sé preciso y no inventes detalles que no conoces."""
    
    # Loop de conversación
    while True:
        pregunta = input("🧑 TÚ: ").strip()
        
        if pregunta.lower() in ['salir', 'exit', 'quit']:
            print("\n👋 ¡Hasta luego!\n")
            break
        
        if not pregunta:
            continue
        
        # Consultar a GPT con configuración anti-alucinación
        try:
            print(f"\n⏳ Consultando {modelo}...\n")
            respuesta = client.chat.completions.create(
                model=modelo,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": pregunta}
                ],
                temperature=0.1,  # Temperatura muy baja para reducir creatividad y alucinaciones
                max_tokens=500,
                top_p=0.9  # Nucleus sampling más restrictivo
            ).choices[0].message.content
            
            print(f"🤖 {modelo.upper()}: {respuesta}\n")
            print("-" * 60 + "\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    main()
