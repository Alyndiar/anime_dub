"""Fonctions utilitaires pour la journalisation et le filtrage des stems.

Ces helpers sont partagés entre les scripts CLI et le GUI afin de
centraliser le mode verbose, la résolution des stems sélectionnés via les
arguments ou les variables d'environnement, et la création de loggers
compatibles avec l'injection par le GUI.
"""

import json
import logging
import os
from typing import Iterable, Optional, Set


def should_verbose(env_value: str | None) -> bool:
    """Retourne True si la valeur d'environnement active le mode verbose."""

    if not env_value:
        return False
    return env_value.lower() in {"1", "true", "yes", "on"}


def init_logger(
    name: str,
    verbose: bool = False,
    external_logger: logging.Logger | None = None,
) -> logging.Logger:
    """Crée ou configure un logger pour un script donné.

    - si ``external_logger`` est fourni (cas du GUI), il est utilisé tel quel ;
    - sinon, un logger local est préparé avec un handler console.
    """

    if external_logger:
        return external_logger

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def parse_stems(
    cli_stems: Iterable[str] | None,
    logger: Optional[logging.Logger] = None,
    env_var: str = "ANIME_DUB_SELECTED_STEMS",
) -> Set[str] | None:
    """Fusionne les stems issus des arguments CLI et de l'environnement.

    :param cli_stems: stems passés via --stem (liste ou None)
    :param logger: logger optionnel pour reporter les erreurs de parsing
    :param env_var: nom de la variable d'environnement à inspecter
    :return: ensemble de stems ou None si aucun élément
    """

    stems: Set[str] = set(cli_stems or [])

    env_raw = os.environ.get(env_var)
    if env_raw:
        try:
            env_values = json.loads(env_raw)
            if isinstance(env_values, str):
                stems.add(env_values)
            elif isinstance(env_values, Iterable):
                stems.update(str(item) for item in env_values)
            else:
                if logger:
                    logger.warning(
                        "Variable %s fournie mais format inattendu (%s)", env_var, type(env_values)
                    )
        except json.JSONDecodeError:
            if logger:
                logger.warning("Impossible de parser %s='%s'", env_var, env_raw)

    return stems or None
