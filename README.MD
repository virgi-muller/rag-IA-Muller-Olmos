# RAG IA - Müller y Olmos

Proyecto de Recuperación Aumentada por Generación (RAG) con Inteligencia Artificial.

## Requisitos previos

- Python 3.10 o superior
- PowerShell (en Windows)

## Configuracion del entorno virtual

### 1. Crear el entorno virtual

```bash
python -m venv venv
```

### 2. Activar el entorno virtual

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

### 3. Instalar dependencias

Con el entorno virtual activado, instala las dependencias del proyecto:

```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno

Copia el archivo de ejemplo y completá las claves necesarias:

```bash
cp .env.example .env
```

Luego editá `.env` con tus credenciales:

| Variable | Descripcion |
|---|---|
| `CHROMA_TENANT` | Tenant de ChromaDB |
| `CHROMA_DATABASE` | Nombre de la base de datos en ChromaDB |
| `CHROMA_API_KEY` | API key de ChromaDB |
| `GOOGLE_API_KEY` | API key de Google (Gemini) |

### 5. Desactivar el entorno virtual

Cuando termines de trabajar:

```bash
deactivate
```
