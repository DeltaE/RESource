docs:
	uv sync --locked --extra docs
	uv run sphinx-build -W --keep-going -b html docs/source docs/_build/html

docs-clean:
	uv sync --locked --extra docs
	uv run sphinx-build -E -a -W --keep-going -b html docs/source docs/_build/html

docs-serve:
	uv run python -m http.server 8000 --directory docs/_build/html

docs-deploy:
	uvx ghp-import --no-jekyll --push --force \
		--branch gh-pages --remote origin docs/_build/html

docs-publish: docs docs-deploy

.PHONY: docs docs-clean docs-serve docs-deploy docs-publish
