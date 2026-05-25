# src/trustworthy_maternal_postpartum_rag/ingestion/document_registry.py

import logging
import hashlib
from pathlib import Path
import yaml

from trustworthy_maternal_postpartum_rag.utils.config import get_config


CFG = get_config("configs/pipeline_config.yaml")

logger = logging.getLogger("tmprag.ingestion.document_registry")

DOCUMENT_REGISTRY_PATH = Path(
    CFG.get("paths", {}).get("document_registry", "configs/document_registry.yml")
)

STRICT_DOCUMENT_REGISTRY = CFG.get("preprocessing", {}).get(
    "strict_document_registry", True
)

_REQUIRED_REGISTRY_FIELDS = [
    "doc_id",
    "file_path",
    "publisher",
    "source_tier",
    "document_style",
    "lifecycle_stage",
    "topic_scope",
    "country_scope",
    "priority_score",
]

_DOCUMENT_REGISTRY_CACHE = None


def _normalize_path_key(path_like) -> str:
    return Path(str(path_like)).as_posix().strip().lower()


def _validate_registry_entry(doc: dict):
    missing = [field for field in _REQUIRED_REGISTRY_FIELDS if field not in doc]

    if missing and STRICT_DOCUMENT_REGISTRY:
        raise ValueError(
            f"Document registry entry is missing required fields {missing}: {doc}"
        )

    if missing:
        logger.warning(
            "[Registry] Entry has missing fields %s. Defaults will be used. Entry=%s",
            missing,
            doc,
        )


def load_document_registry(registry_path: Path = DOCUMENT_REGISTRY_PATH):
    """
    Load document-level metadata from configs/document_registry.yml.

    Expected YAML structure:

    documents:
      - doc_id: who_antenatal_care
        file_path: "data/raw/Pregnancy_postpartum/who_antenatal care.pdf"
        publisher: "WHO"
        source_tier: "core_authoritative"
        document_style: "guideline"
        lifecycle_stage: "pregnancy"
        topic_scope: "antenatal_care"
        country_scope: "global"
        priority_score: 4
        target: "mother"
    """

    if not registry_path.exists():
        msg = (
            f"[Registry] document_registry.yml not found at {registry_path}. "
            "Falling back to filename-based metadata."
        )

        if STRICT_DOCUMENT_REGISTRY:
            raise FileNotFoundError(
                msg + " Because strict_document_registry=True, preprocessing cannot continue."
            )

        logger.warning(msg)
        return {"by_path": {}, "by_name": {}}

    with open(registry_path, "r", encoding="utf-8") as f:
        registry = yaml.safe_load(f) or {}

    documents = registry.get("documents", [])
    registry_by_path = {}
    registry_by_name = {}

    for doc in documents:
        _validate_registry_entry(doc)

        file_path = doc.get("file_path")
        if not file_path:
            continue

        path_key = _normalize_path_key(file_path)
        name_key = Path(file_path).name.lower().strip()

        if path_key in registry_by_path:
            logger.warning("[Registry] Duplicate file_path detected: %s", file_path)

        if name_key in registry_by_name:
            logger.warning("[Registry] Duplicate filename detected: %s", name_key)

        registry_by_path[path_key] = doc
        registry_by_name[name_key] = doc

    logger.info("[Registry] Loaded %d documents from %s", len(documents), registry_path)

    return {
        "by_path": registry_by_path,
        "by_name": registry_by_name,
    }


def get_document_registry():
    global _DOCUMENT_REGISTRY_CACHE

    if _DOCUMENT_REGISTRY_CACHE is None:
        _DOCUMENT_REGISTRY_CACHE = load_document_registry()

    return _DOCUMENT_REGISTRY_CACHE


def get_registry_metadata(pdf_path: Path):
    registry = get_document_registry()

    if not registry:
        return None

    try:
        rel_path = pdf_path.relative_to(Path.cwd())
    except ValueError:
        rel_path = pdf_path

    path_key = _normalize_path_key(rel_path)
    name_key = pdf_path.name.lower().strip()

    meta = registry["by_path"].get(path_key)
    if meta:
        return meta

    meta = registry["by_name"].get(name_key)
    if meta:
        return meta

    if STRICT_DOCUMENT_REGISTRY:
        raise KeyError(
            f"No document_registry.yml entry found for PDF: {pdf_path}. "
            "Add this file to configs/document_registry.yml or set "
            "strict_document_registry: false in configs/pipeline_config.yaml."
        )

    logger.warning(
        "[Registry] No registry entry found for %s. Using fallback filename inference.",
        pdf_path.name,
    )

    return None


def infer_doc_metadata(pdf_name: str):
    """
    Fallback metadata inference from filename.

    Used only when strict_document_registry=False or registry metadata is unavailable.
    Output follows the same canonical metadata schema.
    """

    name = pdf_name.lower().strip()
    name_norm = name.replace("-", "_").replace(" ", "_")

    meta = {
        "doc_title": Path(pdf_name).stem,
        "publisher": "unknown",
        "source_tier": "unknown",
        "document_style": "unknown",
        "lifecycle_stage": "unknown",
        "topic_scope": "unknown",
        "country_scope": "unknown",
        "priority_score": 1,
        "target": "unknown",
    }

    if "who_antenatal_care" in name_norm:
        meta.update({
            "publisher": "WHO",
            "source_tier": "core_authoritative",
            "document_style": "guideline",
            "lifecycle_stage": "pregnancy",
            "topic_scope": "antenatal_care",
            "country_scope": "global",
            "priority_score": 4,
            "target": "mother",
        })

    elif "who_pcpnc_third_edition" in name_norm:
        meta.update({
            "publisher": "WHO",
            "source_tier": "core_authoritative",
            "document_style": "clinical_practice_guide",
            "lifecycle_stage": "pregnancy_childbirth_postpartum_newborn",
            "topic_scope": "pregnancy_childbirth_postpartum_newborn_care",
            "country_scope": "global",
            "priority_score": 4,
            "target": "mother_baby",
        })

    elif "who_postnatal_positive_experience" in name_norm:
        meta.update({
            "publisher": "WHO",
            "source_tier": "core_authoritative",
            "document_style": "guideline",
            "lifecycle_stage": "postpartum_newborn",
            "topic_scope": "postnatal_maternal_newborn_care",
            "country_scope": "global",
            "priority_score": 4,
            "target": "mother_baby",
        })

    elif "india_pmsma" in name_norm or "high_risk_conditions_in_preg" in name_norm:
        meta.update({
            "publisher": "Government of India / PMSMA",
            "source_tier": "core_authoritative",
            "document_style": "public_health_guidance",
            "lifecycle_stage": "pregnancy",
            "topic_scope": "high_risk_pregnancy",
            "country_scope": "India",
            "priority_score": 3,
            "target": "mother",
        })

    elif "nhs" in name_norm and "pregnancy" in name_norm:
        meta.update({
            "publisher": "NHS",
            "source_tier": "core_authoritative",
            "document_style": "patient_friendly",
            "lifecycle_stage": "pregnancy_postpartum",
            "topic_scope": "pregnancy_postpartum_guidance",
            "country_scope": "UK",
            "priority_score": 3,
            "target": "mother_baby",
        })

    elif "newborn_and_children_care" in name_norm and "nhs" in name_norm:
        meta.update({
            "publisher": "NHS",
            "source_tier": "core_authoritative",
            "document_style": "patient_friendly",
            "lifecycle_stage": "newborn_infant_child",
            "topic_scope": "newborn_child_care",
            "country_scope": "UK",
            "priority_score": 3,
            "target": "baby_child",
        })

    elif "baby_411" in name_norm:
        meta.update({
            "publisher": "Baby 411",
            "source_tier": "secondary_patient_friendly",
            "document_style": "commercial_book",
            "lifecycle_stage": "newborn_infant",
            "topic_scope": "baby_care",
            "country_scope": "US",
            "priority_score": 1,
            "target": "baby",
        })

    elif "acog" in name_norm and "pregnancy" in name_norm:
        meta.update({
            "publisher": "ACOG",
            "source_tier": "core_authoritative",
            "document_style": "patient_friendly",
            "lifecycle_stage": "pregnancy",
            "topic_scope": "pregnancy_questions",
            "country_scope": "US",
            "priority_score": 3,
            "target": "mother",
        })

    elif "cleveland_clinic" in name_norm and "pregnancy" in name_norm:
        meta.update({
            "publisher": "Cleveland Clinic",
            "source_tier": "secondary_patient_friendly",
            "document_style": "patient_friendly",
            "lifecycle_stage": "pregnancy",
            "topic_scope": "healthy_pregnancy",
            "country_scope": "US",
            "priority_score": 2,
            "target": "mother",
        })

    return meta


def build_doc_metadata(pdf_path: Path, doc_id: str | None = None):
    """
    Build one canonical document metadata object.

    Canonical metadata fields:
    doc_id, doc_title, source_file, source_path, publisher, source_tier,
    document_style, lifecycle_stage, topic_scope, country_scope,
    priority_score, target, metadata_source, doc_version.
    """

    registry_meta = get_registry_metadata(pdf_path)

    if registry_meta:
        final_doc_id = doc_id or registry_meta.get("doc_id") or hashlib.sha256(
            str(pdf_path).encode("utf-8", errors="ignore")
        ).hexdigest()

        metadata = {
            "doc_id": final_doc_id,
            "doc_title": registry_meta.get("doc_title", Path(pdf_path).stem),
            "source_file": pdf_path.name,
            "source_path": str(pdf_path),
            "publisher": registry_meta.get("publisher", "unknown"),
            "source_tier": registry_meta.get("source_tier", "unknown"),
            "document_style": registry_meta.get("document_style", "unknown"),
            "lifecycle_stage": registry_meta.get("lifecycle_stage", "unknown"),
            "topic_scope": registry_meta.get("topic_scope", "unknown"),
            "country_scope": registry_meta.get("country_scope", "unknown"),
            "priority_score": int(registry_meta.get("priority_score", 1)),
            "target": registry_meta.get("target", "unknown"),
            "metadata_source": "document_registry",
            "doc_version": CFG.get("run", {}).get("version", "v1"),
        }

        return final_doc_id, metadata

    fallback_meta = infer_doc_metadata(pdf_path.name)

    final_doc_id = doc_id or hashlib.sha256(
        str(pdf_path).encode("utf-8", errors="ignore")
    ).hexdigest()

    metadata = {
        "doc_id": final_doc_id,
        "doc_title": fallback_meta.get("doc_title", Path(pdf_path).stem),
        "source_file": pdf_path.name,
        "source_path": str(pdf_path),
        "publisher": fallback_meta.get("publisher", "unknown"),
        "source_tier": fallback_meta.get("source_tier", "unknown"),
        "document_style": fallback_meta.get("document_style", "unknown"),
        "lifecycle_stage": fallback_meta.get("lifecycle_stage", "unknown"),
        "topic_scope": fallback_meta.get("topic_scope", "unknown"),
        "country_scope": fallback_meta.get("country_scope", "unknown"),
        "priority_score": int(fallback_meta.get("priority_score", 1)),
        "target": fallback_meta.get("target", "unknown"),
        "metadata_source": "fallback_filename_inference",
        "doc_version": CFG.get("run", {}).get("version", "v1"),
    }

    return final_doc_id, metadata