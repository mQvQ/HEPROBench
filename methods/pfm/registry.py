from __future__ import annotations


PATHOLOGY_FOUNDATION_MODELS = {
    "hoptimus0": "H-Optimus-0",
    "h0-mini": "H0-mini",
    "ctranspath": "CTransPath",
    "conch": "CONCH",
    "conchv1_5": "CONCH v1.5",
    "uni": "UNI",
    "univ2": "UNI v2",
    "gpfm": "GPFM",
    "phikonv2": "Phikon-v2",
    "pathgen": "PathGen",
    "chief": "CHIEF",
    "keep": "KEEP",
    "virchow2": "Virchow2",
    "omiclip": "OmiCLIP",
    "provgigapath": "Prov-GigaPath",
}


def get_pathology_foundation_model(name: str) -> str:
    key = name.lower()
    if key not in PATHOLOGY_FOUNDATION_MODELS:
        raise KeyError(
            f"Unknown pathology foundation model '{name}'. "
            f"Available: {', '.join(sorted(PATHOLOGY_FOUNDATION_MODELS))}"
        )
    return PATHOLOGY_FOUNDATION_MODELS[key]

