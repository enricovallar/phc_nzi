import json

with open("notebooks/phd/slab_c4v_1.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        src = cell["source"]
        found = False
        for line in src:
            if "p_ij" in line or "fom" in line or 'linearity_metric_v2' in line:
                print(f"Cell {idx}: {line.strip()}")
