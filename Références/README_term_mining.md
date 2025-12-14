# Détection automatique de nouveaux termes récurrents (Douluo Dalu / Soul Land)

Objectif : analyser tes JSON chinois **nettoyés** (Whisper → `clean_zh.json`) sur plusieurs épisodes, afin de produire une liste **priorisée** de *candidats* à ajouter au glossaire maître, avec contextes pour validation rapide.

## Entrées attendues
- Dossier contenant des JSON au format :
  ```json
  {"segments":[{"start":0.0,"end":1.2,"text_zh":"..."}, ...]}
  ```
- Glossaire maître (celui que je t’ai fourni) :
  `douluo_glossary_master_fr.json`

## Sorties
- `candidates.json` : liste des candidats (zh, fréquence, #fichiers, contextes)
- `candidates.csv` : idem pour Excel/LibreOffice (délimiteur `;`)

## Commande typique
```bash
python mine_new_terms.py --input-dir CLEAN_ZH_DIR --glossary douluo_glossary_master_fr.json --out-dir OUT --min-count 8 --min-files 2 --top 500
```

## Comment valider “au minimum”
1) Ouvrir `candidates.csv`.
2) Trier par `files` puis `count`.
3) Pour chaque terme pertinent :
   - décider du `type` (term/title/phrase/name)
   - entrer la traduction FR canon
   - l’ajouter au glossaire maître (JSON/CSV)
4) Relancer la détection : les termes déjà validés disparaissent automatiquement.

## Ajustements utiles
- `--n-min/--n-max` : taille des n-grams (2..6 recommandé)
- `--min-count` : monte si trop de bruit, baisse si tu veux capter tôt
- `--min-files` : augmente pour ne garder que les termes vraiment récurrents
- STOP_TOKENS : petit stoplist dans le script, extensible au fil des retours

## Pourquoi ça marche bien pour une série longue
Les termes canon (rangs, techniques, factions, lieux, formules) se répètent. La métrique `files` (présent dans combien d’épisodes) est un excellent signal pour limiter la validation manuelle.
