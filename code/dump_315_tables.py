"""Dump every error/reference table (xlsx via openpyxl, legacy xls via xlrd)
from the 2024.3.15 / 3.24 session folders into CSVs for cross-checking."""
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.expanduser("~/vsip/pylibs"))
from openpyxl import load_workbook
import xlrd

BASE = os.path.expanduser("~/vsip/data/s315")
OUT = os.path.expanduser("~/vsip/results/ref_315")
os.makedirs(OUT, exist_ok=True)


def rows_xlsx(p):
    wb = load_workbook(p, data_only=True)
    out = []
    for ws in wb.worksheets:
        rows = [list(r) for r in ws.iter_rows(values_only=True)]
        while rows and all(v is None for v in rows[-1]):
            rows.pop()
        out.append((ws.title, rows))
    return out


def rows_xls(p):
    out = []
    try:
        book = xlrd.open_workbook(p)
        for ws in book.sheets():
            rows = [[ws.cell_value(r, c) if c < ws.ncols else ""
                     for c in range(ws.ncols)] for r in range(ws.nrows)]
            while rows and all(str(v) == "" for v in rows[-1]):
                rows.pop()
            out.append((ws.name, rows))
    except Exception as e:
        out.append(("ERR", [["open failed", str(e)]]))
    return out


n = 0
for p in sorted(glob.glob(os.path.join(BASE, "*", "*", "*.xls*")) +
                glob.glob(os.path.join(BASE, "*", "误差分析", "*.xls*"))):
    rel = os.path.relpath(p, BASE).replace("\\", "/").replace(" ", "_")
    stem = os.path.splitext(os.path.basename(rel))[0]
    sheets = rows_xlsx(p) if p.endswith("x") else rows_xls(p)
    for title, rows in sheets:
        fn = os.path.join(OUT, f"{stem}__{title.replace('/', '_')}.csv")
        with open(fn, "w", newline="") as f:
            w = csv.writer(f)
            for r in rows:
                w.writerow(["" if v is None else v for v in r])
        n += 1
        # one-line preview of numeric content
        flat = [v for r in rows for v in r if isinstance(v, (int, float))]
        print(f"{os.path.basename(fn):60s} rows={len(rows):3d} nums={len(flat):3d} "
              + ("vals=" + ",".join(f"{v:.4g}" for v in flat[:8]) if flat else ""))
print(f"\n{n} csv files -> {OUT}")
