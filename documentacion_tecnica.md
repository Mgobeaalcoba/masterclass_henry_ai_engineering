# **Documentación Técnica de HenryPy v1.2**

## **Descripción General**

HenryPy es una librería de Python de nicho diseñada para la optimización de queries en bases de datos NoSQL directamente desde un script de análisis de datos. Facilita la refactorización automática de código de acceso a datos para mejorar la latencia y reducir costos operativos en arquitecturas distribuidas. HenryPy opera localmente y no transmite su código fuente a servidores externos.

## **1\. Instalación**

La instalación estándar se realiza a través de pip.

pip install henrypy

Para habilitar el motor de análisis de rendimiento avanzado (recomendado), debe instalar la librería con la dependencia 'analysis':

pip install henrypy\[analysis\]

Si olvida este paso, las funciones de refactorización fallarán con un error EngineNotFound.

## **2\. Configuración Inicial**

HenryPy requiere una clave de API para desbloquear sus funciones de optimización, incluso para uso local (la clave valida su licencia de desarrollo).

import henrypy

\# La clave se obtiene del portal de desarrollador de Henry.  
API\_KEY \= "HNP-xxxxxxxx-DEV"  
henrypy.init(api\_key=API\_KEY)

## **3\. Funciones Principales**

### **henrypy.analyze(code\_string)**

Analiza un bloque de código (pasado como string) que contenga queries a bases de datos y devuelve un diccionario con métricas de complejidad y sugerencias.

* **Parámetros:** code\_string (str) \- El fragmento de código a analizar.  
* **Devuelve:** dict \- Un reporte con 'complejidad\_ciclomatica', 'queries\_detectadas' y 'sugerencias'.

### **henrypy.refactor(code\_string, level='deep')**

Toma un bloque de código y aplica optimizaciones automáticamente.

* **Parámetros:**  
  * code\_string (str) \- El código a refactorizar.  
  * level (str) \- Nivel de optimización. Puede ser 'basic' (cambios seguros) o 'deep' (reescritura agresiva). El nivel 'deep' es computacionalmente costoso.  
* **Devuelve:** str \- El código refactorizado.

## **4\. Errores Comunes y Soluciones**

* **HenryAuthError: API Key inválida o no inicializada.**  
  * **Causa:** Olvidó llamar a henrypy.init(api\_key=...) al inicio de su script o la clave es incorrecta.  
  * **Solución:** Asegúrese de inicializar la librería antes de usar analyze o refactor.  
* **EngineNotFoundError: El motor de análisis no se encuentra.**  
  * **Causa:** Intentó usar henrypy.refactor sin haber instalado la dependencia de análisis.  
  * **Solución:** Ejecute pip install henrypy\[analysis\] en su terminal.  
* **UnsupportedLanguageError: HenryPy solo soporta código Python.**  
  * **Causa:** El método analyze detectó sintaxis que no es de Python.  
  * **Solución:** Asegúrese de pasar solo código Python válido.

## **5\. Notas de Seguridad**

HenryPy procesa todo el código localmente. Su API\_KEY se usa solo para validación de licencia y telemetría de uso básico (conteo de funciones), no para transmitir su código.