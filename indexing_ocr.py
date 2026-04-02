import os
import sys
import warnings
import logging
import base64

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
from mistralai.client import Mistral
from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

load_dotenv()

collection_name = "rag-muller-olmos-ocr"


# ------------------------------------------------------------
# CUSTOM LOADER: Mistral OCR
# ------------------------------------------------------------

class MistralOCRLoader(BaseLoader):
    """DocumentLoader que usa la API OCR de Mistral para extraer texto
    de un PDF local como markdown estructurado."""

    def __init__(self, file_path: str, api_key: str):
        self.file_path = file_path
        self.api_key = api_key

    def load(self) -> list[Document]:
        client = Mistral(api_key=self.api_key)

        with open(self.file_path, "rb") as f:
            pdf_data = base64.standard_b64encode(f.read()).decode("utf-8")

        response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{pdf_data}",
            },
        )

        documents = []
        for i, page in enumerate(response.pages):
            documents.append(Document(
                page_content=page.markdown,
                metadata={
                    "source": self.file_path,
                    "page": i + 1,
                },
            ))

        return documents


# ------------------------------------------------------------
# CARGAR EL MODELO DE EMBEDDINGS
# ------------------------------------------------------------

print("\n Loading embedding model...")

embeddings = MistralAIEmbeddings(
    model="mistral-embed"
)

print("[INFO] Embedding model loaded successfully")
print("[INFO] Model: mistral-embed (Mistral AI)")


# ------------------------------------------------------------
# 1, 2 y 4. CARGAR DATOS Y CREAR BASE VECTORIAL (solo si vacía)
# ------------------------------------------------------------

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

    api_key = os.getenv("MISTRAL_API_KEY")

    # PASO 1 — Cargar archivos vía Mistral OCR
    print("\n[PASO 1] Extrayendo texto con Mistral OCR desde la carpeta 'data'...")

    data_dir = "data"
    raw_documents = []

    for file_path in glob.glob(os.path.join(data_dir, "*")):
        if not os.path.isfile(file_path):
            continue
        try:
            if file_path.endswith('.pdf'):
                loader = MistralOCRLoader(file_path, api_key=api_key)
            elif file_path.endswith('.txt'):
                from langchain_community.document_loaders import TextLoader
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_path.endswith('.csv'):
                from langchain_community.document_loaders import CSVLoader
                loader = CSVLoader(file_path)
            else:
                print(f"  [WARNING] Formato no soportado: {file_path}")
                continue

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

    # PASO 2 — Dividir en fragmentos con MarkdownHeaderTextSplitter
    print("\n[PASO 2] Dividiendo documentos en fragmentos (chunks)...")

    headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=True,
    )

    # Agrupar páginas por documento fuente y splitear el markdown completo
    from collections import defaultdict
    pages_by_source: dict[str, list[Document]] = defaultdict(list)
    for doc in raw_documents:
        pages_by_source[doc.metadata["source"]].append(doc)

    md_splits: list[Document] = []
    for source, pages in pages_by_source.items():
        full_markdown = "\n\n".join(p.page_content for p in pages)
        splits = md_splitter.split_text(full_markdown)
        for doc in splits:
            doc.metadata["source"] = source
        md_splits.extend(splits)

    # Prepend de headers al page_content para que el embedding capture la jerarquía
    for doc in md_splits:
        prefix = " > ".join(
            doc.metadata[k] for k in ("H1", "H2", "H3") if k in doc.metadata
        )
        if prefix:
            doc.page_content = f"{prefix}\n{doc.page_content}"

    # Segundo paso: cortar chunks que superen el límite
    second_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    documents = second_splitter.split_documents(md_splits)

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
