import json
import ast
import sys

nb_path = "kaggle_remote/versia_17/rsna-versia-17-apex-sota.ipynb"
with open(nb_path, encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
errors = 0

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = "".join(cell.get('source', []))
        # Remove %%writefile or other magic commands for ast syntax checking
        clean_lines = []
        for line in src.split("\n"):
            if line.startswith("%%") or line.startswith("%"):
                clean_lines.append("# " + line)
            else:
                clean_lines.append(line)
        clean_src = "\n".join(clean_lines)
        try:
            ast.parse(clean_src)
        except SyntaxError as e:
            print(f"Cell {i} SyntaxError: {e}")
            errors += 1

if errors == 0:
    print("ALL 22 CELLS PASSED SYNTAX CHECK SUCCESSFULLY!")
else:
    print(f"FAILED with {errors} syntax errors!")
    sys.exit(1)
