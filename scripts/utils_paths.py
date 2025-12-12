"""Utilitaires de nommage pour harmoniser les stems sur disque.

- accepte des stems avec espaces en entrée (GUI/CLI)
- normalise les noms de fichiers en remplaçant les espaces par des underscores
"""

from __future__ import annotations

from typing import Iterable, Set


def normalize_stem(stem: str) -> str:
    """Remplace les espaces par des underscores pour un nom de fichier sûr."""

    return stem.replace(" ", "_")


def normalized_filter(stems_filter: Iterable[str] | None) -> Set[str] | None:
    """Retourne l'ensemble des stems normalisés pour filtrage, si fourni."""

    if stems_filter is None:
        return None
    return {normalize_stem(stem) for stem in stems_filter}


def stem_matches_filter(stem: str, normalized_filter_set: Set[str] | None) -> bool:
    """Vérifie qu'un stem correspond au filtre (en version normalisée)."""

    if normalized_filter_set is None:
        return True
    return normalize_stem(stem) in normalized_filter_set
