# Douluo Dalu / Soul Land – Glossaire maître + QA (FR)

## Fichiers
- `douluo_glossary_master_fr.json` : glossaire maître (pipeline)
- `douluo_glossary_master_fr.csv`  : version tableur
- `douluo_glossary_master_fr.md`   : version documentation
- `qa_validate_glossary.py`        : validation/enforcement local après traduction

## Format JSON attendu (exemple)
```json
{"segments":[{"start":0.0,"end":1.2,"text_zh":"武魂","text_fr":"Esprit martial"}]}
```

## QA – rapport uniquement
```bash
python qa_validate_glossary.py in_fr.json douluo_glossary_master_fr.json report.json
```

## QA – enforcement (réparation best-effort)
```bash
python qa_validate_glossary.py in_fr.json douluo_glossary_master_fr.json report.json --mode enforce --out out_fr_fixed.json
```

## Notes importantes
- Les entrées `type=name` sont traitées comme **non traduisibles** (jamais remplacées, uniquement détectées/ajoutées).
- L’enforcement actuel est volontairement conservateur (append/insert). Tu peux durcir (replace) si tu le souhaites.
