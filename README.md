# Chat con tu PDF

## Cómo ejecutar el proyecto localmente

### Opción recomendada: Docker

1. Asegúrate de tener Docker Desktop instalado y en ejecución.
2. Crea el archivo `.env` en la raíz del proyecto. Puedes usar `.env.example` como base.
3. Coloca tu clave de OpenAI en `OPENAI_API_KEY`.
4. Levanta la aplicación con:

```bash
docker compose up --build
```

5. Abre la app en el navegador:

```text
http://127.0.0.1:8000
```

### Archivo `.env`

El proyecto usa variables de entorno para la configuración. Las más importantes son:

```env
OPENAI_API_KEY=tu_clave_aqui
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini
TOP_K_RESULTS=4
MIN_RELEVANCE_SCORE=0.35
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
TEMPERATURE=0
MAX_UPLOAD_MB=20
```

Si prefieres no usar Docker, también puedes correrlo con Python localmente:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Cómo usar la aplicación

1. En la interfaz web, selecciona un archivo PDF.
2. Haz clic en `Cargar PDF` para procesarlo.
3. Espera a que el sistema extraiga el texto, cree los chunks y genere los embeddings.
4. Escribe una pregunta sobre el contenido del PDF.
5. Presiona `Enviar pregunta`.
6. Revisa:
   - la respuesta del chatbot
   - los chunks recuperados
   - el visor del PDF en la columna izquierda

Notas de uso:

- La aplicación trabaja con un solo PDF a la vez.
- Si subes otro PDF, reemplaza el anterior.
- Si el documento no tiene texto extraíble, se mostrará un error claro.
- Si la pregunta no tiene suficiente relación con el PDF, el sistema responderá de forma prudente y no inventará información.

## Descripción breve del proyecto

Este proyecto es una aplicación web tipo “Chat with your PDF” construida con FastAPI, HTML/CSS/JS y OpenAI. Permite subir un solo PDF, extraer su texto, dividirlo en chunks, generar embeddings, guardar los vectores en FAISS y responder preguntas usando solo el contexto recuperado.

La interfaz está pensada para una entrega universitaria: simple, clara, en español y fácil de demostrar en video.

## Arquitectura del sistema

La arquitectura es sencilla y está dividida en tres partes:

- `Backend`: FastAPI maneja las rutas, la carga del PDF, el procesamiento y las preguntas.
- `Servicio RAG`: extrae texto, hace chunking, crea embeddings, arma FAISS, recupera contexto y genera respuestas con OpenAI.
- `Frontend`: HTML/CSS/JS renderizado desde FastAPI muestra el visor del PDF, el historial del chat, la respuesta y los chunks recuperados.

El estado del documento se guarda en memoria dentro de la aplicación, por lo que no se usa base de datos.

## Tecnologías usadas

- FastAPI
- Jinja2
- HTML
- CSS
- JavaScript
- PyPDF
- LangChain
- RecursiveCharacterTextSplitter
- OpenAI Embeddings
- FAISS
- python-dotenv
- Docker
- docker-compose

## Flujo del sistema

```text
PDF → extracción de texto → chunks → embeddings → vector store FAISS → retrieval top-k → respuesta del LLM
```

Flujo detallado:

1. El usuario sube un PDF desde la interfaz.
2. El backend valida que el archivo sea PDF.
3. Se extrae el texto con `PyPDF`.
4. El texto se divide en chunks con `RecursiveCharacterTextSplitter`.
5. Cada chunk se convierte en embeddings con OpenAI.
6. Los embeddings se guardan en un vector store local con FAISS.
7. Cuando el usuario hace una pregunta, el sistema recupera los chunks más relevantes con búsqueda semántica top-k.
8. Se construye un prompt en español con el contexto recuperado.
9. El LLM de OpenAI responde usando solo ese contexto.
10. La interfaz muestra la respuesta final y los chunks recuperados.

## Estructura del proyecto

```text
app/
  main.py
  services/
    config.py
    pdf_service.py
    rag_service.py
  static/
    app.js
    style.css
  templates/
    index.html
  uploads/
requirements.txt
Dockerfile
docker-compose.yml
.env.example
.env
.dockerignore
.gitignore
README.md
```

## Manejo básico de errores

El proyecto incluye validaciones y mensajes en español para estos casos:

- archivo no subido
- archivo que no es PDF
- PDF vacío o dañado
- pregunta antes de procesar un PDF
- falta de `OPENAI_API_KEY`
- error de API al generar embeddings o respuestas
- similitud insuficiente para responder con confianza

Cuando no hay suficiente contexto, el sistema no inventa información y responde de forma prudente.

## Posibles mejoras futuras

- permitir historial persistente entre recargas
- soportar múltiples PDF con selección por documento
- agregar resaltado de fragmentos relevantes dentro del visor
- permitir limpiar o reemplazar el PDF desde un botón dedicado
- mostrar progreso visual durante el procesamiento
- exportar preguntas y respuestas a un archivo
- agregar pruebas automatizadas para backend y RAG
