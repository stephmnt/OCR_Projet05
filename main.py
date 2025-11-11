from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

PIPELINE_STEPS = [
    ("Préparation des données brutes", "projet_05.dataset"),
    ("Feature engineering", "projet_05.features"),
    ("Entraînement du modèle", "projet_05.modeling.train"),
]


def run_step(label: str, module_path: str) -> None:
    """Execute one stage of the training pipeline.

    Args:
        label: Human‑readable step name for logging.
        module_path: Dotted path to the module to execute (used with `python -m`).

    Raises:
        RuntimeError: If the subprocess exits with a non‑zero status.
    """
    logger.info("➡️  Étape '{}' en cours...", label)
    completed = subprocess.run(
        [sys.executable, "-m", module_path],
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:
        logger.error("Échec pour '{}'.", label)
        if completed.stdout:
            logger.error("STDOUT:\n{}", completed.stdout)
        if completed.stderr:
            logger.error("STDERR:\n{}", completed.stderr)
        raise RuntimeError(f"L'étape '{label}' a échoué (code {completed.returncode}).")

    if completed.stdout:
        logger.debug(completed.stdout.strip())
    logger.success("Étape '{}' terminée.", label)


def main() -> None:
    """Run all pipeline stages sequentially."""
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{timestamp}.log"

    logger.add(log_file, level="INFO", enqueue=True)
    logger.info("Début d'exécution du pipeline (log: {})", log_file)

    for label, module in PIPELINE_STEPS:
        run_step(label, module)

    logger.success("Pipeline exécuté avec succès. Logs disponibles dans {}", log_file)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - orchestration script
        logger.error("Pipeline interrompu : {}", exc)
        sys.exit(1)
