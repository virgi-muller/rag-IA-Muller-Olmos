import os
import sys
import warnings
import logging

warnings.filterwarnings("ignore")

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
os.environ["POSTHOG_DISABLED"] = "1"

_SILENT = logging.CRITICAL + 1

logging.root.setLevel(_SILENT)
for _h in logging.root.handlers[:]:
    logging.root.removeHandler(_h)
logging.root.addHandler(logging.NullHandler())

for _noisy in ("pypdf", "pypdf._cmap", "pypdf._page",
               "langchain", "langchain_core", "chromadb"):
    logging.getLogger(_noisy).setLevel(_SILENT)

import glob
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

collection_name = "rag-muller-olmos"

# ------------------------------------------------------------
# CARGAR EL MODELO DE EMBEDDINGS
# ------------------------------------------------------------
# Se carga siempre, tanto para crear la DBV como para consultar.

print("\n Loading embedding model...")

embeddings = MistralAIEmbeddings(
    model="mistral-embed"
)

print("[INFO] Embedding model loaded successfully")
print("[INFO] Model: mistral-embed (Mistral AI)")


# ------------------------------------------------------------
# 1, 2 y 4. CARGAR DATOS Y CREAR BASE VECTORIAL (solo si vacía)
# ------------------------------------------------------------
# Si la colección ya tiene documentos se reutiliza directamente,
# evitando re-indexar todos los documentos en cada ejecución.
# La conexión a Chroma Cloud lee CHROMA_API_KEY, CHROMA_TENANT
# y CHROMA_DATABASE del .env a través de langchain-chroma.

vectorstore = Chroma(
    collection_name=collection_name,
    embedding_function=embeddings,
    chroma_cloud_api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
)

if vectorstore._collection.count() > 0:
    # ── Camino rápido: cargar la colección ya indexada ───────────
    print(f"\n{'─'*60}")
    print(f"[INFO] Colección '{collection_name}' encontrada en Chroma Cloud")
    print(f"[INFO] Documentos existentes: {vectorstore._collection.count()}")
    print(f"[INFO] Cargando colección existente... (se omite re-indexación)")
    print(f"{'─'*60}")

    print("[OK]   Vector database cargada correctamente")
    print(f"[OK]   Colección: {collection_name}")

else:
    # ── Carga, división e indexación ───────
    print(f"\n{'─'*60}")
    print(f"[AVISO] Colección '{collection_name}' vacía. Construyendo desde cero...")
    print(f"{'─'*60}")

    # PASO 1 — Cargar archivos
    print("\n[PASO 1] Cargando archivos desde la carpeta 'data'...")

    data_dir = "data"
    raw_documents = []

    for file_path in glob.glob(os.path.join(data_dir, "*")):
        if not os.path.isfile(file_path):
            continue
        try:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.csv'):
                loader = CSVLoader(file_path)
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                print(f"  [WARNING] Formato no soportado: {file_path}")
                continue

            # Suprimir salida de librerías ruidosas durante la carga
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
            try:
                docs = loader.load()
            finally:
                sys.stdout.close()
                sys.stderr.close()
                sys.stdout, sys.stderr = old_stdout, old_stderr

            raw_documents.extend(docs)
            print(f"  [OK] {len(docs):>4} páginas/docs  ←  {os.path.basename(file_path)}")
        except Exception as e:
            print(f"  [ERROR] No se pudo cargar {file_path}: {e}")

    print(f"[INFO] Total de documentos crudos cargados: {len(raw_documents)}")

    # PASO 2 — Dividir en fragmentos (chunks)
    print("\n[PASO 2] Dividiendo documentos en fragmentos (chunks)...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    documents = text_splitter.split_documents(raw_documents)

    print(f"[INFO] Fragmentos (chunks) generados: {len(documents)}")
    print(f"[INFO] Tamaño de chunk: 800 chars | Overlap: 150 chars")

    # PASO 4 — Crear e indexar en Chroma Cloud (en batches de 300)
    BATCH_SIZE = 300
    batches = [documents[i:i + BATCH_SIZE] for i in range(0, len(documents), BATCH_SIZE)]

    print("\n[PASO 4] Indexando en Chroma Cloud...")
    print(f"[INFO] Colección: {collection_name}")
    print(f"[INFO] Batches: {len(batches)} × {BATCH_SIZE} docs máx.")

    # Primer batch: crea la colección
    vectorstore = Chroma.from_documents(
        batches[0],
        embedding=embeddings,
        collection_name=collection_name,
        chroma_cloud_api_key=os.getenv("CHROMA_API_KEY"),
        tenant=os.getenv("CHROMA_TENANT"),
        database=os.getenv("CHROMA_DATABASE"),
    )
    print(f"  [OK] Batch 1/{len(batches)}  ({len(batches[0])} chunks)")

    # Batches restantes: añade a la colección existente
    for idx, batch in enumerate(batches[1:], start=2):
        vectorstore.add_documents(batch)
        print(f"  [OK] Batch {idx}/{len(batches)}  ({len(batch)} chunks)")

    print("[OK]   Base vectorial creada en Chroma Cloud")
    print(f"[OK]   Colección: {collection_name}")
    print(f"[OK]   Total de chunks indexados: {len(documents)}")
    print(f"{'─'*60}")
