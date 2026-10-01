# datasets

Reserved for versioned evaluation datasets and their dataset cards. The first real dataset is planned for M4, so this directory is still empty. The loader (`niriksha.core.dataset.load_dataset`) expects a directory containing `dataset.json` and `cases.jsonl`; the only datasets in the repository today are synthetic test fixtures under `tests/fixtures/datasets/`.

Each dataset needs a card based on [docs/datasets/_dataset-card-template.md](../docs/datasets/_dataset-card-template.md). `datasets/raw/` is gitignored; do not commit downloaded data whose licence has not been recorded.
