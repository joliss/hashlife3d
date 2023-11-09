## Installing dependencies

```
python3 -m venv .venv
source .venv/bin/activate
pip install poetry
poetry install
pip install -e .
```

## Running

```
python -m hashlife3d ~/src/golly/Patterns/HashLife/Metacell/metapixel-p216-gun.mc.gz output.mp4
```

assuming your golly clone is in `~/src/golly`.
