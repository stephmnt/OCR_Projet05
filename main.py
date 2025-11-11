from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent

PIPELINE_STEPS = [
    ("Préparation des données brutes", REPO_ROOT / "projet_05" / "dataset.py"),
    ("Feature engineering", REPO_ROOT / "projet_05" / "features.py"),
    ("Entraînement du modèle", REPO_ROOT / "projet_05" / "modeling" / "train.py"),
]


def run_step(label: str, script_path: Path) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"Script introuvable : {script_path}")

    logger.info("➡️  Étape '{}' en cours...", label)
    completed = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:
        logger.error("❌ Échec pour '{}'.", label)
        if completed.stdout:
            logger.error("STDOUT:\n{}", completed.stdout)
        if completed.stderr:
            logger.error("STDERR:\n{}", completed.stderr)
        raise RuntimeError(f"L'étape '{label}' a échoué (code {completed.returncode}).")

    if completed.stdout:
        logger.debug(completed.stdout.strip())
    logger.success("✅ Étape '{}' terminée.", label)


def main() -> None:
    for label, path in PIPELINE_STEPS:
        run_step(label, path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - orchestration script
        logger.error("Pipeline interrompu : {}", exc)
        sys.exit(1)
