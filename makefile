###############################################################################
# Makefile –  make      → venv + editable install + Snakemake
#             make clean→ wipe venv + artefacts + data
###############################################################################
REQUIRED_PYTHON := python3.11
PYTHON ?= $(REQUIRED_PYTHON)

VENV   := .venv            # DO NOT leave trailing spaces here!

# helper paths
PY  := $(strip $(VENV))/bin/$(PYTHON)
PIP := $(strip $(VENV))/bin/pip

# default goal ---------------------------------------------------------------
all: check-python-version venv install deps resource


# 1) venv --------------------------------------------------------------------
$(VENV):
	$(PYTHON) -m venv $@

venv: | $(VENV)

# 2) editable install --------------------------------------------------------
install: venv setup.py
	$(PIP) install --upgrade pip
	$(PIP) install -e .

# 3) dependencies ------------------------------------------------------------
deps: venv requirements.txt
	$(PIP) install -r requirements.txt

# 4) run the pipeline --------------------------------------------------------
resource:
	$(PYTHON) run.py
# housekeeping ---------------------------------------------------------------
clean:
	rm -rf $(strip $(VENV))/ build dist *.egg-info data_scraping data/*.csv || true

.PHONY: all venv install deps scrape clean
###############################################################################
check-python-version:
	@version=$$($(PYTHON) -c 'import sys; print(".".join(map(str, sys.version_info[:2])))'); \
	if [ "$$version" != "3.11" ]; then \
	  echo "❌ Python 3.11 is required. Found $$version."; exit 1; \
	fi
