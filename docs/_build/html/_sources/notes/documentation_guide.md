# Documentation Build Guide

This guide covers building, developing, and deploying RESource documentation using Sphinx.

## 🚀 Quick Start

### Build Documentation Locally
```bash
# Build documentation (includes notebook sync)
make docs-build

# View locally
open docs/build/html/index.html  # macOS
xdg-open docs/build/html/index.html  # Linux
```

### Live Development
```bash
# Start live rebuild server (recommended for development)
make autobuild

# Opens http://localhost:8000 with auto-reload
# Changes to source files automatically rebuild and refresh browser
```

---

## 📖 Documentation Commands

### Building
```bash
make docs-build        # Build documentation only
make sync-notebooks    # Sync notebooks from root to docs/source/notebooks/
make docs-conda        # Build and deploy to GitHub Pages
```

### Development
```bash
make autobuild         # Live rebuild with auto-reload server
make clean-docs        # Clean build files
```

### Deployment
```bash
make deploy            # Deploy to GitHub Pages
```

---

## 📁 Documentation Structure

```
docs/
├── source/                    # Source files
│   ├── index.md              # Main documentation page
│   ├── conf.py               # Sphinx configuration
│   ├── _static/              # Static assets (images, CSS)
│   ├── _templates/           # Custom templates
│   ├── notes/                # Documentation pages
│   │   ├── setup_guide.md    # Deprecated (see SETUP.md in root)
│   │   ├── api.md            # API documentation
│   │   ├── developers.md     # Developer guide
│   │   └── ...
│   └── notebooks/            # Jupyter notebooks (synced from root)
│       ├── Store_explorer.ipynb
│       ├── Visuals_BC.ipynb
│       └── resource_module_runner.ipynb
├── build/                     # Generated HTML (auto-created)
│   └── html/                 # HTML output
└── requirements.txt          # Documentation dependencies
```

---

## 🔧 Sphinx Configuration

### Key Settings (docs/source/conf.py)
The documentation uses:
- **Theme:** `sphinx-book-theme` (modern, responsive)
- **Format:** MyST Markdown (`.md` files with extended syntax)
- **Extensions:** 
  - `myst_parser` - Markdown support
  - `nbsphinx` - Jupyter notebook integration
  - `sphinx.ext.autodoc` - API documentation
  - `sphinx.ext.viewcode` - Source code links

### Dependencies
Documentation dependencies are managed through the main conda environment:
```yaml
# In env/environment.yml
- sphinx
- pip:
  - myst-parser
  - sphinx-book-theme  
  - nbsphinx
  - sphinx-autobuild
  - ghp-import
```

---

## 📓 Notebook Integration

### Automatic Sync
The `make docs-build` command automatically syncs notebooks:

**Source Locations → Documentation:**
- `notebooks/Store_explorer.ipynb` → `docs/source/notebooks/Store_explorer.ipynb`
- `notebooks/Visuals_BC.ipynb` → `docs/source/notebooks/Visuals_BC.ipynb`  
- `resource_module_runner.ipynb` → `docs/source/notebooks/resource_module_runner.ipynb`

### Manual Sync
```bash
make sync-notebooks
```

### Adding New Notebooks
1. Add notebook to root `notebooks/` directory
2. Update sync command in Makefile:
   ```makefile
   sync-notebooks:
       @cp notebooks/your_new_notebook.ipynb docs/source/notebooks/
   ```
3. Add to `docs/source/index.md` toctree:
   ```markdown
   ```{toctree}
   :caption: 'Notebooks:'
   
   notebooks/your_new_notebook
   ```

---

## 🌐 GitHub Pages Deployment

### Automated Deployment
```bash
make deploy
```

This command:
1. Builds documentation (`make docs-build`)
2. Creates Jekyll bypass files (`.nojekyll`, `_config.yml`)
3. Deploys to `gh-pages` branch using `ghp-import`
4. Documentation available at: https://deltae.github.io/RESource/

### Manual GitHub Pages Setup
If setting up for first time:
1. Go to repository Settings → Pages
2. Set source to "Deploy from a branch"
3. Select branch: `gh-pages`
4. Path: `/ (root)`

### Jekyll Bypass
The build process creates these files to bypass Jekyll processing:
- `.nojekyll` - Disables Jekyll
- `_config.yml` - Minimal Jekyll config
- `README.md` - GitHub Pages notice

---

## ✍️ Writing Documentation

### MyST Markdown Syntax
Documentation uses MyST (Markedly Structured Text) which extends Markdown:

**Basic MyST:**
```markdown
# Page Title

## Section

Regular markdown content...

```{note}
This is a note box
```

```{warning}
This is a warning box
```
```

**Cross-references:**
```markdown
[Setup Guide](../../../SETUP.md)  # Link to consolidated setup guide
{ref}`section-label`      # Link to section
```

**Code blocks with syntax highlighting:**
```markdown
```python
import pandas as pd
df = pd.DataFrame({'col': [1, 2, 3]})
```
```

### Adding New Documentation Pages
1. Create `.md` file in `docs/source/notes/`
2. Add to `docs/source/index.md` toctree:
   ```markdown
   ```{toctree}
   :caption: 'Contents:'
   
   notes/your_new_page
   ```

---

## 🔍 Development Workflow

### Live Development (Recommended)
```bash
# Start live reload server
make autobuild

# In another terminal, edit files in docs/source/
# Browser automatically refreshes on changes
# Available at: http://localhost:8000
```

### Build and Check
```bash
# Build documentation
make docs-build

# Check for warnings/errors in output
# View locally: docs/build/html/index.html
```

### Deploy Changes  
```bash
# Build and deploy to GitHub Pages
make deploy

# View live: https://deltae.github.io/RESource/
```

---

## 🐛 Troubleshooting

### Common Issues

**Build Errors:**
```bash
# Check environment has Sphinx dependencies
make conda-status

# Clean build and retry
make clean-docs
make docs-build
```

**Notebook Issues:**
```bash
# Ensure notebooks are synced
make sync-notebooks

# Check notebook paths in index.md toctree
```

**GitHub Pages Not Updating:**
- Check gh-pages branch exists and has content
- Verify GitHub Pages settings point to gh-pages branch
- Check Actions tab for deployment status

**Live Reload Not Working:**
- Ensure port 8000 is available
- Check for firewall issues
- Try different port: `sphinx-autobuild docs/source docs/build --port 8001`

### Debug Commands
```bash
# Verbose build
sphinx-build -v -b html docs/source docs/build/html

# Check configuration
python -c "import sphinx; print(sphinx.__version__)"

# List extensions
python -c "from docs.source.conf import extensions; print(extensions)"
```

---

## 📋 Documentation Checklist

### Before Committing
- [ ] Documentation builds without errors: `make docs-build`
- [ ] Live reload works: `make autobuild`
- [ ] New notebooks are synced: `make sync-notebooks`
- [ ] Cross-references work
- [ ] Images and assets are included

### Before Release
- [ ] All documentation is current
- [ ] API documentation is updated
- [ ] Examples and tutorials work
- [ ] Deploy to GitHub Pages: `make deploy`
- [ ] Verify live site: https://deltae.github.io/RESource/

---

## 💡 Best Practices

### Content Organization
- Keep setup/installation info in `SETUP.md` (repository root)
- API documentation in `api.md`
- Developer guides in `developers.md`
- Use descriptive filenames without spaces

### Writing Style
- Use MyST admonitions (`{note}`, `{warning}`) for callouts
- Include code examples with proper syntax highlighting
- Cross-reference other documentation sections
- Keep line length reasonable for editing

### Maintenance
- Update documentation with code changes
- Test all code examples
- Keep screenshot/images current
- Regular spelling/grammar checks

---

*This guide covers all aspects of RESource documentation development and deployment.*
