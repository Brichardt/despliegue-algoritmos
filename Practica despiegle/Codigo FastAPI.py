from fastapi import FastAPI
from transformers import pipeline

# Creación de la aplicación FastAPI
app = FastAPI()

# Pipeline de Hugging Face para análisis de sentimiento
analizador_sentimientos = pipeline("sentiment-analysis")

# Pipeline de Hugging Face para generación de texto
generador_texto = pipeline("text-generation", model="gpt2")

# Módulo 1 : Endpoint principal
@app.get("/")
def leer_inicio():
    return {"mensaje": "Bienvenido a la API FastAPI con Hugging Face!"}

# Módulo 2 : Análisis de sentimiento
@app.get("/sentimiento/{texto}")
def analizar_sentimiento(texto: str):
    resultado = analizador_sentimientos(texto)
    return {"entrada": texto, "sentimiento": resultado}

# Módulo 3 : Contar las vocales
@app.get("/contar_vocales/{texto}")
def contar_vocales(texto: str):
    vocales = "aeiouAEIOU"
    count = sum(1 for char in texto if char in vocales)
    return {"texto": texto, "numero_vocales": count}


# Módulo 4 : Retornar el inverso de un texto
@app.get("/invertir/{texto}")
def invertir_texto(texto: str):
    texto_invertido = texto[::-1]
    return {"original": texto, "invertido": texto_invertido}

# Módulo 5 : Número de palabras en una frase
@app.get("/contar_palabras/{texto}")
def contar_palabras(texto: str):
    conteo = len(texto.split())
    return {"texto": texto, "conteo_palabras": conteo}

# Ejecutar el servidor con uvicorn si el script se ejecuta directamente
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
