import os
import json
import argparse
import warnings
import logging
import time

warnings.filterwarnings("ignore")

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
os.environ["POSTHOG_DISABLED"] = "1"

_SILENT = logging.CRITICAL + 1

logging.root.setLevel(_SILENT)
for _h in logging.root.handlers[:]:
    logging.root.removeHandler(_h)
logging.root.addHandler(logging.NullHandler())

for _noisy in ("langchain", "langchain_core", "langchain_mistralai",
               "chromadb", "httpx", "httpcore"):
    logging.getLogger(_noisy).setLevel(_SILENT)

from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ------------------------------------------------------------

QUESTIONS_FILE  = "questions.json"
OUTPUT_ANSWERS  = "results_answers.json"
OUTPUT_FULL     = "results_full.json"

COLLECTION_STANDARD = "standard_collection"
COLLECTION_OCR      = "ocr_collection"

# Las 3 configuraciones a comparar:
#   config_1_standard : colección extracción estándar (PyPDFLoader), top_k=5
#   config_2_ocr      : colección extracción OCR + markdown chunking,  top_k=5
#   config_3_topk     : colección estándar con top_k=10 (doble contexto)
CONFIGS = [
    {"name": "config_1_standard", "collection": COLLECTION_STANDARD, "top_k": 5},
    {"name": "config_2_ocr",      "collection": COLLECTION_OCR,      "top_k": 5},
    {"name": "config_3_topk",     "collection": COLLECTION_STANDARD, "top_k": 10},
]

# ------------------------------------------------------------
# PROMPT DEL LLM
# Instrucción explícita: si el contexto no es suficiente → decir que no sabe.
# ------------------------------------------------------------

SYSTEM_PROMPT = (
    "Eres un asistente especializado en documentación médica. "
    "Responde ÚNICAMENTE basándote en el contexto proporcionado a continuación. "
    "Si el contexto no contiene información suficiente o relevante para responder "
    "la pregunta, responde exactamente: "
    "\"No lo sé. La información no está disponible en los documentos consultados.\" "
    "No inventes información ni utilices conocimiento externo al contexto dado."
)

HUMAN_PROMPT = (
    "CONTEXTO:\n{context}\n\n"
    "PREGUNTA:\n{question}\n\n"
    "RESPUESTA:"
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_PROMPT),
])

# ------------------------------------------------------------
# INICIALIZAR EMBEDDINGS Y LLM
# ------------------------------------------------------------

print(f"\n{'─'*60}")
print("[INIT] Inicializando modelo de embeddings (mistral-embed)...")
embeddings = MistralAIEmbeddings(model="mistral-embed")
print("[OK]   Embeddings listos")

print("[INIT] Inicializando LLM (open-mistral-nemo)...")
llm = ChatMistralAI(model="open-mistral-nemo", temperature=0.0)
chain = prompt_template | llm
print("[OK]   LLM listo")
print(f"{'─'*60}")

# ------------------------------------------------------------
# CONECTAR A LAS COLECCIONES EN CHROMA CLOUD
# ------------------------------------------------------------

def connect_collection(collection_name: str) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        chroma_cloud_api_key=os.getenv("CHROMA_API_KEY"),
        tenant=os.getenv("CHROMA_TENANT"),
        database=os.getenv("CHROMA_DATABASE"),
    )

print("[INIT] Conectando a colecciones en Chroma Cloud...")

vectorstores: dict[str, Chroma] = {}
for config in CONFIGS:
    cname = config["collection"]
    if cname not in vectorstores:
        print(f"  → Conectando '{cname}'...")
        vectorstores[cname] = connect_collection(cname)
        print(f"  [OK] '{cname}' conectada")

print(f"{'─'*60}")

# ------------------------------------------------------------
# FUNCIÓN PRINCIPAL DE PROCESAMIENTO
# ------------------------------------------------------------

def process_question(q_id, q_text) -> tuple[dict, dict]:
    print(f"\n{'='*60}")
    print(f"  Pregunta #{q_id}: {q_text}")
    print(f"{'='*60}")

    answer_row = {"id": q_id, "question": q_text}
    full_row   = {"id": q_id, "question": q_text, "configurations": {}}

    for config in CONFIGS:
        cfg_name   = config["name"]
        collection = config["collection"]
        top_k      = config["top_k"]
        vs         = vectorstores[collection]

        print(f"\n  [{cfg_name}]  colección='{collection}'  top_k={top_k}")

        # ── Recuperación vectorial ──────────────────────────────
        docs = vs.similarity_search(q_text, k=top_k)
        print(f"  [OK] {len(docs)} chunk(s) recuperados")

        # ── Construcción del contexto ───────────────────────────
        context_text = "\n\n".join(doc.page_content for doc in docs)

        # ── Generación de respuesta ────────────────────────────
        response = chain.invoke({"context": context_text, "question": q_text})
        answer   = response.content.strip()

        print(f"  [OK] Respuesta generada ({len(answer)} chars)")
        
        # Retardo para evitar el límite de peticiones de la API gratuita
        time.sleep(2)

        # ── Serializar chunks para el JSON completo ─────────────
        chunks_serialized = []
        for rank, doc in enumerate(docs, start=1):
            chunks_serialized.append({
                "rank":    rank,
                "content": doc.page_content,
                "source":  os.path.basename(doc.metadata.get("source", "desconocido")),
                "page":    doc.metadata.get("page", None),
            })

        # ── Acumular resultados ─────────────────────────────────
        answer_row[cfg_name] = answer

        full_row["configurations"][cfg_name] = {
            "collection": collection,
            "top_k":      top_k,
            "answer":     answer,
            "chunks":     chunks_serialized,
        }

    return answer_row, full_row


# ------------------------------------------------------------
# MODO INDIVIDUAL: imprimir respuestas en terminal
# ------------------------------------------------------------

def print_individual_results(answer_row: dict) -> None:
    print(f"\n{'─'*60}")
    for config in CONFIGS:
        cfg_name = config["name"]
        print(f"\n  [{cfg_name}]")
        print(f"  {answer_row[cfg_name]}")
    print(f"\n{'─'*60}")


# ------------------------------------------------------------
# PUNTO DE ENTRADA
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG – recuperación y generación")
    parser.add_argument(
        "--mode", choices=["batch", "individual"], default="batch",
        help="'batch' lee questions.json (default) | 'individual' pide pregunta por terminal",
    )
    parser.add_argument(
        "--question", "-q", type=str, default=None,
        help="Pregunta directa (solo en modo individual)",
    )
    args = parser.parse_args()

    if args.mode == "individual":
        # ── Modo individual ─────────────────────────────────────
        q_id = 0
        first = True
        while True:
            if first and args.question:
                q_text = args.question.strip()
            else:
                try:
                    q_text = input("\nPregunta (o 'salir' para terminar): ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n[Saliendo]")
                    break

            first = False

            if q_text.lower() in ("salir", "exit", "quit", ""):
                print("[Saliendo]")
                break

            q_id += 1
            answer_row, _ = process_question(q_id, q_text)
            print_individual_results(answer_row)

            # Si la pregunta vino por argumento, no hacer loop
            if args.question:
                break

    else:
        # ── Modo batch (comportamiento original) ────────────────
        print(f"[LOAD] Leyendo preguntas desde '{QUESTIONS_FILE}'...")

        with open(QUESTIONS_FILE, encoding="utf-8") as f:
            questions_data = json.load(f)

        questions = questions_data["questions"]
        print(f"[OK]   {len(questions)} pregunta(s) cargada(s)")
        print(f"{'─'*60}")

        results_answers = []
        results_full    = []

        for q_item in questions:
            answer_row, full_row = process_question(q_item["id"], q_item["question"])
            results_answers.append(answer_row)
            results_full.append(full_row)

        print(f"\n{'─'*60}")
        print(f"[SAVE] Guardando '{OUTPUT_ANSWERS}'...")
        with open(OUTPUT_ANSWERS, "w", encoding="utf-8") as f:
            json.dump(results_answers, f, ensure_ascii=False, indent=2)
        print(f"[OK]   {OUTPUT_ANSWERS} guardado")

        print(f"[SAVE] Guardando '{OUTPUT_FULL}'...")
        with open(OUTPUT_FULL, "w", encoding="utf-8") as f:
            json.dump(results_full, f, ensure_ascii=False, indent=2)
        print(f"[OK]   {OUTPUT_FULL} guardado")

        print(f"{'─'*60}")
        print(f"\n[LISTO] Procesadas {len(questions)} pregunta(s) × {len(CONFIGS)} configuraciones.")
        print(f"  → Respuestas:     {OUTPUT_ANSWERS}")
        print(f"  → Detalle chunks: {OUTPUT_FULL}")
        print(f"{'─'*60}\n")