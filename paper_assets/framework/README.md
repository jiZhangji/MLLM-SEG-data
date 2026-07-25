# FreeRef Framework Figure

This directory contains the deterministic, code-generated main framework
figure for the FreeRef paper. Every visual element is produced by Matplotlib
and NumPy; the script does not depend on the reference screenshot or external
image assets.

## Generate

From the repository root:

```bash
python -m pip install -r paper_assets/framework/requirements.txt
python paper_assets/framework/generate_framework_figure.py
```

To write directly into a LaTeX paper directory:

```bash
python paper_assets/framework/generate_framework_figure.py \
  --output-dir /path/to/paper/figures
```

The script emits:

- `freeref_framework.pdf`: paper-ready vector output;
- `freeref_framework.svg`: editable vector source;
- `freeref_framework.png`: 300-DPI preview.
- `freeref_framework_components/`: the generated input scene, probability
  fields, intervention fields, hard mask, SLIC view, regional fields, and
  changed-pixel visualization used inside the framework figure.

No photograph or intermediate illustration needs to be downloaded. The demo
scene and every component image are generated deterministically by the same
script.

The three panels match the method definition used in the paper:

1. soft/hard outputs are adapted to the common `(p, u)` interface;
2. semantic anchors and an image-aware region graph define a sparse SPD solve;
3. uncertainty-gated pixel reconstruction enforces the intervention bound.
