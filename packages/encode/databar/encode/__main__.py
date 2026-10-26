# Technique: a __main__.py makes the package runnable as `python -m databar.encode`.
# It stays tiny — just import the real entry point and call it.
from .cli import main

main()
