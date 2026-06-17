#!/usr/bin/env python
"""Download + prepare the FEP+ JACS benchmark targets into the RBFE_Example layout.

For each target it writes  data/<target>/{protein.pdb, ligands.sdf, edges.yml,
ligands.yml}.

Source: OpenFE benchmarks (OpenFreeEnergy/openfe-benchmarks, jacs_set). This is
the FULL Wang-2015 JACS set -- ligand counts match the literature exactly
(cdk2=16, mcl1=42, p38=34, ptp1b=23, thrombin=11, tyk2=16, bace=36, jnk1=21) --
with curated **Kartograf** atom mappings and experimental dG bundled in one
`industry_benchmarks_network.json` (an OpenFE LigandNetwork / GUFE graphml):
  * each node's molprops carries `ofe-name` (ligand name) + `r_exp_dg` (kcal/mol),
  * each edge's `mapping` is the Kartograf atom map [[A_idx, B_idx], ...] over the
    ligands.sdf atom order.

We parse that into edges.yml (with atom mappings, same schema as the rest of the
pipeline) and ligands.yml (name -> dg_exp_kcal). protein.pdb and ligands.sdf are
copied verbatim.

Usage:  python prepare_targets.py [target ...]   (default: all 8)
"""
import json
import os
import sys
import urllib.request
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
GN = "{http://graphml.graphdrawing.org/xmlns}"
RAW = ("https://raw.githubusercontent.com/OpenFreeEnergy/openfe-benchmarks/"
       "main/openfe_benchmarks/data/benchmark_systems/jacs_set/{t}/{f}")
TARGETS = ["bace", "cdk2", "jnk1", "mcl1", "p38", "ptp1b", "thrombin", "tyk2"]


def _get(target, fname):
    with urllib.request.urlopen(RAW.format(t=target, f=fname), timeout=90) as r:
        return r.read()


def parse_network(net_bytes):
    """OpenFE LigandNetwork json -> (name->dg, [(a, b, {A_idx: B_idx}), ...])."""
    graphml = json.loads(net_bytes.decode())[0][1]["graphml"]
    root = ET.fromstring(graphml)                 # raw (entity-escaped) -> ET unescapes
    keys = {k.get("attr.name"): k.get("id") for k in root.iter(GN + "key")}
    node_key, map_key = keys["moldict"], keys["mapping"]

    id2name, dg = {}, {}
    for node in root.iter(GN + "node"):
        for d in node:
            if d.get("key") == node_key:
                props = json.loads(d.text).get("molprops", {}) or {}
                id2name[node.get("id")] = props.get("ofe-name")
                if props.get("r_exp_dg") is not None:
                    dg[props["ofe-name"]] = props["r_exp_dg"]
    edges = []
    for e in root.iter(GN + "edge"):
        amap = None
        for d in e:
            if d.get("key") == map_key:
                amap = {int(a): int(b) for a, b in json.loads(d.text)}
        edges.append((id2name[e.get("source")], id2name[e.get("target")], amap))
    return dg, edges


def prepare(target):
    tdir = os.path.join(HERE, target)
    os.makedirs(tdir, exist_ok=True)
    print(f"[{target}]")
    for f in ("protein.pdb", "ligands.sdf"):
        print(f"    {f}")
        with open(os.path.join(tdir, f), "wb") as fh:
            fh.write(_get(target, f))

    dg, edges = parse_network(_get(target, "industry_benchmarks_network.json"))
    # Quote names: many targets use purely-numeric ligand IDs (e.g. "23470"),
    # which YAML would otherwise read back as ints and fail to match the string
    # ligand names in the SDF.
    with open(os.path.join(tdir, "ligands.yml"), "w") as fh:
        for name, val in dg.items():
            fh.write(f'"{name}":\n  name: "{name}"\n  dg_exp_kcal: {val}\n')
    n = 0
    with open(os.path.join(tdir, "edges.yml"), "w") as fh:
        fh.write("remarks: OpenFE benchmarks jacs_set, Kartograf atom mappings\n"
                 "edges:\n")
        for a, b, amap in edges:
            if a is None or b is None or not amap:
                continue
            fh.write(f"  edge_{a}_{b}:\n    ligand_a: \"{a}\"\n    ligand_b: \"{b}\"\n")
            fh.write("    atom mapping: {"
                     + ", ".join(f"{k}: {v}" for k, v in amap.items()) + "}\n")
            n += 1
    print(f"    {len(dg)} ligands, {n} edges (Kartograf maps)")


def main():
    for t in sys.argv[1:] or TARGETS:
        if t not in TARGETS:
            print(f"unknown target {t!r}; known: {TARGETS}")
            continue
        prepare(t)
    print("done.")


if __name__ == "__main__":
    main()
