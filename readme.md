`uv init`
`uv venv --seed`
`.venv\Scripts\activate`
`uv sync --extra cuda`
`python setup_blender.py`

```
sitplus/
├── pyproject.toml
├── sitplus/
│   ├── __init__.py
│   ├── generators/
│   │   ├── __init__.py
│   │   └── parametric.py
│   └── utils/
│       ├─ __init__.py
│       ├─ attention.py
│       ├─ dwt_tokenizer.py
│       ├─ encoder.py
│       └─ masks.py
│
└── tests/
    ├── test_wavelet_core.py
    ├── some
    ├── other
    ├── stuff
    └── i guess ...?
```