import os

matches = []
for root, dirs, files in os.walk("."):
    if ".next" in root or "node_modules" in root or ".git" in root or "brain" in root:
        continue
    for f in files:
        if f.endswith((".py", ".ts", ".tsx", ".js")):
            p = os.path.join(root, f)
            try:
                content = open(p, encoding="utf-8").read()
                if "m_nz_ind_2026_t20_1" in content:
                    print(f"Found in {p}")
            except Exception:
                pass
