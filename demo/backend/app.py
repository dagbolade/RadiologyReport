"""
FastAPI backend for the Radiology Report Generator demo.
DevBcn Barcelona 2026 - Live Demo Endpoint

Endpoints:
  GET  /              → serve frontend
  GET  /health        → detailed health status
  POST /api/analyze   → upload X-ray, get report + attention map
  POST /api/translate → translate report text
  GET  /api/info      → model architecture and research info
"""

import os
import re
import time
import uuid
import logging
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from model import RadiologyModel, compute_severity

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("radiology-api")

# ─── Model singleton ────────────────────────────────────────────────────────

radiology_model = RadiologyModel()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup"""
    logger.info("Starting model loading...")
    start = time.time()
    try:
        radiology_model.load()
        elapsed = time.time() - start
        logger.info(f"Model loaded in {elapsed:.1f}s")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    yield
    logger.info("Shutting down...")


# ─── App ─────────────────────────────────────────────────────────────────────

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

app = FastAPI(
    title="Radiology Report Generator API",
    description="Deep Learning for Medical Report Generation - DevBcn 2026 Demo",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Text post-processing ───────────────────────────────────────────────────

def clean_report_text(raw_text: str) -> str:
    """Clean up model output: fix spacing, capitalize sentences."""
    text = raw_text.strip()
    # Remove special tokens
    text = text.replace('<cls>', '').replace('<end>', '').replace('<unk>', '').strip()
    # Fix spacing before punctuation: "word ." → "word."
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    # Capitalize first letter of each sentence
    text = re.sub(r'(^|[.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
    # Capitalize first character
    if text:
        text = text[0].upper() + text[1:]
    return text


# ─── Plain English interpretation ───────────────────────────────────────────

# Pattern-based interpreter: matches common radiology report phrases and
# produces a contextual plain-English reading of the *whole impression*,
# not just term definitions.

PHRASE_PATTERNS = [
    # ── Normal / clear findings ──────────────────────────────────────────
    (r"no acute cardiopulmonary abnormality",
     "The heart and lungs look normal — no signs of any new or sudden problems."),
    (r"no acute disease\b",
     "No signs of any new or sudden illness were found on the X-ray."),
    (r"no acute findings",
     "Nothing unusual or concerning was detected on this X-ray."),
    (r"(lungs are|lung fields are|lungs appear)\s*(clear|normal|unremarkable)",
     "The lungs appear healthy and clear, with no visible signs of disease."),
    (r"(heart size|cardiac silhouette)\s*(is )?(normal|within normal limits|unremarkable)",
     "The heart appears to be a normal size."),
    (r"within normal limits",
     "Everything on this X-ray falls within the normal range."),

    # ── Cardiomegaly ─────────────────────────────────────────────────────
    (r"(stable )?cardiomegaly without superimposed.*(acute|disease)",
     "The heart is larger than normal (a condition called cardiomegaly), but this appears to be a known, stable condition — there are no new complications or sudden problems on top of it."),
    (r"stable cardiomegaly",
     "The heart is enlarged, but this hasn't changed compared to previous exams — it's a known condition that is staying the same."),
    (r"cardiomegaly",
     "The heart appears larger than normal on the X-ray. This could be caused by various conditions such as high blood pressure, heart valve disease, or heart failure, and may need further evaluation."),

    # ── Atelectasis / collapse ───────────────────────────────────────────
    (r"low lung volumes?.*(atelectasis|bibasilar)",
     "The lungs are not fully expanded. There are areas at the base of both lungs where the tissue has partially collapsed or folded in on itself, which can happen after surgery, from shallow breathing, or when lying down for extended periods."),
    (r"bibasilar (atelectasis|opacit)",
     "There are changes visible at the bottom of both lungs — most likely areas where the lung tissue has partially collapsed. This is common and can occur from shallow breathing or prolonged bed rest."),
    (r"atelectasis",
     "Part of the lung tissue has partially collapsed or is not fully inflated. This is relatively common and can happen after surgery, during illness, or from shallow breathing."),

    # ── Opacities / infiltrates ──────────────────────────────────────────
    (r"(opacit|infiltrat).*(pulmonary edema|edema).*difficult to.*(exclude|entirely exclude)",
     "There are cloudy areas on the X-ray that could indicate either partially collapsed lung tissue or fluid buildup in the lungs. The model cannot fully rule out fluid in the lungs (pulmonary edema), which sometimes occurs with heart problems."),
    (r"(bibasilar|bilateral) opacit",
     "There are cloudy or hazy areas visible at the base of both lungs. This could be caused by fluid, infection, inflammation, or partially collapsed lung tissue."),
    (r"patchy opacit",
     "There are scattered cloudy patches visible in the lungs, which could indicate infection, inflammation, or fluid buildup."),
    (r"opacit",
     "There are cloudy or whitish areas visible on the X-ray, which could indicate fluid, infection, or other changes in the lung tissue."),
    (r"infiltrat",
     "There are signs that fluid or inflammatory material has accumulated in parts of the lung tissue, which can happen with infections like pneumonia."),

    # ── Pulmonary edema ──────────────────────────────────────────────────
    (r"pulmonary edema",
     "There are signs of fluid buildup in the lungs, which is often related to heart problems. This can cause shortness of breath and typically requires medical attention."),

    # ── Pleural effusion ─────────────────────────────────────────────────
    (r"(bilateral|small) pleural effusion",
     "There is fluid collecting in the space around the lungs. This can be caused by infections, heart failure, or other conditions and may need to be drained if significant."),
    (r"pleural effusion",
     "Fluid has accumulated in the space between the lung and the chest wall, which can cause breathing difficulties."),

    # ── Pneumonia / consolidation ────────────────────────────────────────
    (r"consolidation",
     "An area of the lung appears solid or filled with fluid instead of air, which is often a sign of pneumonia or infection."),
    (r"pneumonia",
     "There are signs consistent with a lung infection (pneumonia), where parts of the lung have become inflamed and filled with fluid."),

    # ── Devices / post-surgical ──────────────────────────────────────────
    (r"sternotomy",
     "There are signs of previous heart surgery visible (surgical wires in the breastbone)."),
    (r"pacemaker",
     "A pacemaker device is visible on the X-ray, indicating the patient has an implanted heart rhythm device."),
    (r"catheter",
     "A medical tube (catheter) is visible on the X-ray, which is a device used for treatment or monitoring."),

    # ── Degenerative / spine ─────────────────────────────────────────────
    (r"degenerative changes",
     "There are signs of normal age-related wear and tear, most likely in the spine or joints."),
    (r"scoliosis",
     "The spine shows an abnormal sideways curvature."),

    # ── Emphysema / COPD ─────────────────────────────────────────────────
    (r"(hyperinflat|emphysema)",
     "The lungs appear over-inflated, which is often seen in chronic lung conditions like emphysema or COPD. The air sacs in the lungs may be damaged, making it harder to breathe."),

    # ── Nodule / mass ────────────────────────────────────────────────────
    (r"(pulmonary )?nodule",
     "A small round spot was found in the lung. Most lung nodules are harmless, but follow-up imaging may be recommended to monitor it over time."),
    (r"\bmass\b",
     "An abnormal growth or lump was detected, which will need further investigation to determine its nature."),

    # ── Bronchovascular ──────────────────────────────────────────────────
    (r"bronchovascular crowding",
     "The airways and blood vessels in the lungs appear closer together than normal, which happens when the lungs aren't fully expanded."),
    (r"bronchovascular",
     "Changes are visible in the airways and blood vessels within the lungs."),

    # ── Low lung volumes ─────────────────────────────────────────────────
    (r"low lung volume",
     "The lungs are not as fully expanded as they could be, which can happen from shallow breathing, pain, or lying flat during the X-ray."),
]


def generate_plain_english(report: str) -> str:
    """
    Interpret a radiology report impression into plain English.
    Uses pattern matching to understand the *meaning* of the full report,
    not just define individual terms.
    """
    text = report.lower().strip()
    if not text:
        return "No report was generated."

    interpretations = []
    used_patterns = set()

    for pattern, explanation in PHRASE_PATTERNS:
        if re.search(pattern, text) and explanation not in used_patterns:
            interpretations.append(explanation)
            used_patterns.add(explanation)
            # Stop after 3 key interpretations to keep it readable
            if len(interpretations) >= 3:
                break

    if not interpretations:
        return ("The X-ray findings are within normal expectations. "
                "No significant abnormalities were identified.")

    # Join into a cohesive paragraph
    result = " ".join(interpretations)

    return result


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Serve the frontend"""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"name": "Radiology Report Generator API", "version": "2.0.0", "docs": "/docs"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": radiology_model._loaded,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/analyze")
async def analyze_xray(
    image1: UploadFile = File(..., description="Primary chest X-ray (frontal view)"),
    image2: UploadFile = File(None, description="Optional second X-ray (lateral view)"),
    patient_name: str = Form(""),
    patient_age: str = Form(""),
    patient_gender: str = Form(""),
):
    """
    Analyze chest X-ray(s) and generate a radiology report.
    Accepts optional patient metadata for the report.
    """
    if not radiology_model._loaded:
        raise HTTPException(status_code=503, detail="Model is still loading. Please try again shortly.")

    allowed_types = {"image/jpeg", "image/png", "image/jpg", "application/octet-stream"}
    if image1.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {image1.content_type}. Accepted: JPEG, PNG")

    try:
        start_time = time.time()

        image1_bytes = await image1.read()
        image2_bytes = await image2.read() if image2 else None

        # Generate report
        result = radiology_model.generate_report(image1_bytes, image2_bytes)

        # Clean up report text
        cleaned_report = clean_report_text(result["report"])

        # Generate plain English explanation
        plain_english = generate_plain_english(cleaned_report)

        # Generate attention visualization
        attention_image = radiology_model.generate_attention_map(
            image1_bytes, result["attention_weights"]
        )

        # Compute severity
        severity = compute_severity(cleaned_report)

        elapsed = time.time() - start_time

        return {
            "report_id": str(uuid.uuid4()),
            "generated_report": cleaned_report,
            "plain_english": plain_english,
            "attention_visualization": attention_image,
            "severity": severity,
            "inference_time_seconds": round(elapsed, 2),
            "num_tokens_generated": result["num_tokens"],
            "timestamp": datetime.now().isoformat(),
            "patient": {
                "name": patient_name or "Anonymous",
                "age": patient_age or "—",
                "gender": patient_gender or "—",
            },
            "model_info": {
                "name": "Attention-CheXNet",
                "encoder": "DenseNet-121 (CheXNet)",
                "framework": "TensorFlow/Keras",
            },
            "disclaimer": "AI-generated report for research purposes. Not for clinical diagnosis.",
        }

    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/translate")
async def translate_report(
    text: str = Form(...),
    target_language: str = Form(...),
):
    """Translate report text to target language using deep-translator."""
    logger.info(f"Translation request: lang={target_language}, text_len={len(text)}")
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source='auto', target=target_language).translate(text)
        return {"translated_text": translated, "target_language": target_language}
    except ImportError:
        raise HTTPException(status_code=501, detail="Translation service not available. Install deep-translator.")
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


@app.get("/api/info")
async def model_info():
    """Return model architecture and research information"""
    return {
        "model": {
            "name": "Attention-CheXNet Report Generator",
            "architecture": {
                "encoder": "DenseNet-121 (CheXNet pre-trained on ChestX-ray14)",
                "attention": "Bahdanau (additive) global attention",
                "decoder": "GRU with greedy search decoding",
                "embedding": "300-dim (GloVe initialized)",
                "input_size": "224x224 RGB",
                "max_tokens": 28,
            },
            "performance": {
                "bleu_4": 0.482,
                "rouge_l": 0.718,
                "inference_target": "<60 seconds",
                "training_data": "~7,400 chest X-ray image-report pairs",
            },
            "training": {
                "dataset": "Indiana University Chest X-ray Collection",
                "training_samples": 4530,
                "test_samples": 565,
                "class_imbalance": "81% normal cases",
                "framework": "TensorFlow/Keras 2.15",
            },
        },
        "research": {
            "author": "David Agbolade",
            "role": "Senior Data Scientist, UK Government (Ofsted)",
            "publications": [
                {"title": "JIWE (Journal of Informatics and Web Engineering)", "detail": "Vol 5 No 1, February 2026"},
                {"title": "Q1 Springer Publication", "detail": "Referenced citation"},
                {"title": "Scienmag Science Magazine", "detail": "Featured article"},
            ],
            "awards": [
                {"title": "Best Presenter Award", "event": "NexSymp 2025", "location": "Malaysia"},
            ],
            "presentations": [
                {"event": "Builders Foundry London", "date": "April 2026"},
                {"event": "DevBcn Barcelona", "date": "June 2026"},
            ],
        },
    }


# ─── Static files mount (MUST be after all routes) ─────────────────────────
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
