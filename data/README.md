# Supplied data locations

The original files remain in their supplied locations:

- Housing CSV: `housing_baseline_data/hdb-price-prediction-assignment-main/hdb_price_prediction.csv`, relative to the assignment root.
- X-ray archive: `chestxray_binary_64.npz`, relative to the assignment root.
- Housing reference code and lock: `housing_baseline_data/hdb-price-prediction-assignment-main/`.

Paths are recorded in `configs/project.json`; use `pathlib` to resolve them from the assignment root. The setup script records SHA-256 hashes and metadata for reproducibility. Supplied files remain unchanged.

Part A's prepared datasets are saved in `part_a/outputs/data/`, with separate `selection/` (2017–2020 fitting) and `refit/` (2017–2021 fitting) directories. Each has `preprocessing.pt` and `datasets.pt`. The root contains `report.json` and `split_membership.csv`; the latter retains original zero-based CSV row IDs, including excluded 2023 rows for provenance. Dataset tensors contain no 2023 samples and no transformation fits test data.

See [docs/HOUSING_DATA.md](../docs/HOUSING_DATA.md) for the housing schema, API, preparation/verification commands, and baseline-parity evidence. Completed X-ray preparation is in `part_b/outputs/data/`; [Part B preparation](../docs/PART_B_PREPARATION.md) documents preserved partitions, normalization, duplicates and raw overlays. Both standalone submission ZIPs include their respective unchanged raw source and prepared data.
