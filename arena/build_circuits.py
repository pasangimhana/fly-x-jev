#!/usr/bin/env python3
"""Extract real fly-brain circuits from the local MaleCNS v1.0 connectome.

Needs pyarrow + pandas + numpy. Input feathers are read from the MaleCNS
directory, chosen in this order: --malecns-dir PATH, the MALECNS_DIR env var,
else ./male-cns-data at the repo root (see arena/DATA.md for the files):

    python3 arena/build_circuits.py [--malecns-dir PATH]

Writes into ./data/:
    brain_positions.f32   little-endian float32, N x 3 (Three.js coords)
    brain_superclass.u8   uint8, N (index into brain_meta.superclasses)
    brain_meta.json       count, superclasses, bounds, units_note, transform
    circuits.json         odor / MB / DAN / action / looming circuits, all
                          neuron references are indices into the N arrays
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"
DEFAULT_SRC = HERE.parent / "male-cns-data"


def malecns_dir() -> Path:
    """--malecns-dir flag, else $MALECNS_DIR, else <repo>/male-cns-data.

    Parsed at import time so build_vision.py (which imports these constants)
    accepts the same flag.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--malecns-dir")
    args, _ = parser.parse_known_args()
    raw = args.malecns_dir or os.environ.get("MALECNS_DIR")
    return Path(raw).expanduser().resolve() if raw else DEFAULT_SRC


SRC = malecns_dir()


def require_inputs(*paths: Path) -> None:
    missing = [p for p in paths if not p.is_file()]
    if missing:
        names = "\n  ".join(p.name for p in missing)
        sys.exit(f"MaleCNS input not found in {SRC}:\n  {names}\n"
                 "Download it (see arena/DATA.md) or pass --malecns-dir / set MALECNS_DIR.")
ANNOTATIONS = SRC / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
EDGES = SRC / "connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather"

SEED = 7                   # odor glomerulus draw
N_GLOMERULI = 6            # per odor
KC_SPARSITY = 0.05         # APL-like k-winners-take-all
MAX_HOPS = 3
PATHS_PER_ACTION = 8
PATHS_PER_LOOM_TYPE = 4
MAX_PATHS_PER_TERMINAL = 2
BEAM = 4000
TARGET_EXTENT = 2.0        # largest axis spans this many Three.js units
VOXEL_NM = 8.0             # MaleCNS voxel size (nm), for units_note only
MOTOR_SUPERCLASSES = ("vnc_motor", "cb_motor")

T0 = time.time()
NOTES: list[str] = []


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def read_feather(path: Path, columns: list[str]) -> pa.Table:
    with pa.memory_map(str(path), "r") as src:
        return ipc.open_file(src).read_all().select(columns)


# ---------------------------------------------------------------------------
# 1. Neurons + positions
# ---------------------------------------------------------------------------

def load_neurons():
    cols = ["bodyId", "type", "class", "superclass", "status", "somaLocation",
            "tosomaLocation", "somaSide", "rootSide", "flywireType", "synonyms"]
    tab = read_feather(ANNOTATIONS, cols)
    tab = tab.filter(pc.is_in(tab["status"], value_set=pa.array(["Traced", "Anchor"])))
    df = tab.to_pandas()
    n_traced = len(df)
    all_type_counts = df["type"].value_counts().to_dict()
    all_type_counts["__KC_TOTAL__"] = int((df["class"] == "Kenyon_Cell").sum())

    # somaLocation first, tosomaLocation as fallback (same as serve_connectome_3d.py)
    coords = []
    for a, b in zip(df["somaLocation"], df["tosomaLocation"]):
        c = a if a is not None else b
        ok = c is not None and len(c) == 3 and all(v is not None for v in c)
        coords.append(c if ok else None)
    has = np.array([c is not None for c in coords])
    df = df[has].reset_index(drop=True)
    xyz = np.array([coords[i] for i in np.flatnonzero(has)], dtype=np.float64)
    n_fallback = int(sum(1 for a, h in zip(tab["somaLocation"].to_pylist(), has) if h and a is None))

    for col in ("type", "class", "superclass", "somaSide", "rootSide", "flywireType", "synonyms"):
        df[col] = df[col].astype(object).where(df[col].notna(), None)
    df["superclass"] = df["superclass"].fillna("unclassified")

    log(f"{n_traced:,} Traced/Anchor neurons; {len(df):,} with a position "
        f"({n_fallback} via tosomaLocation fallback); {n_traced - len(df):,} dropped (no soma coord)")
    NOTES.append(
        f"Positions are soma coordinates (somaLocation, falling back to tosomaLocation for {n_fallback} cells). "
        f"{n_traced - len(df):,} of {n_traced:,} Traced/Anchor neurons have neither (mostly sensory neurons whose "
        "somata lie outside the CNS) and are excluded from every array and every path.")
    return df, xyz, all_type_counts


def orient(xyz: np.ndarray, df: pd.DataFrame):
    """Map MaleCNS voxel axes to Three.js (y-up, brain on top, VNC below).

    Checked against annotations: image +x -> fly's LEFT (somaSide L),
    image +y -> VENTRAL (SEZ motor neurons > pars intercerebralis),
    image +z -> POSTERIOR (KC somata > ALPN/PAM somata; VNC at large z).
    The VNC sits posterior (and a little ventral) to the brain, so a rigid
    'brain above VNC' layout puts the brain's anterior pole at +y.

      three.x = -(img.x - c)   fly's right -> +x   (left/right optic lobes on x)
      three.y = -(img.z - c)   anterior up, VNC hangs down
      three.z = -(img.y - c)   dorsal faces +z (default camera sees a dorsal view)
    Proper rotation (det = +1): no mirroring.
    """
    centroid = xyz.mean(axis=0)
    span = xyz.max(axis=0) - xyz.min(axis=0)
    scale = TARGET_EXTENT / span.max()
    c = xyz - centroid
    out = np.stack([-c[:, 0], -c[:, 2], -c[:, 1]], axis=1) * scale
    out = out.astype("<f4")

    # sanity: orientation checks
    sc = df["superclass"].values
    side = df["somaSide"].values
    vnc = np.array([s.startswith("vnc_") for s in sc])
    brain = np.array([s.startswith("cb_") or s.startswith("ol_") for s in sc])
    olL = (sc == "ol_intrinsic") & (side == "L")
    olR = (sc == "ol_intrinsic") & (side == "R")
    checks = {
        "vnc_mean_y": float(out[vnc, 1].mean()),
        "brain_mean_y": float(out[brain, 1].mean()),
        "ol_left_mean_x": float(out[olL, 0].mean()),
        "ol_right_mean_x": float(out[olR, 0].mean()),
    }
    assert checks["vnc_mean_y"] < checks["brain_mean_y"], checks
    assert checks["ol_left_mean_x"] < 0 < checks["ol_right_mean_x"], checks
    log(f"orientation ok: VNC mean y {checks['vnc_mean_y']:.2f} < brain {checks['brain_mean_y']:.2f}; "
        f"left OL x {checks['ol_left_mean_x']:.2f}, right OL x {checks['ol_right_mean_x']:.2f}")
    return out, centroid, scale, checks


# ---------------------------------------------------------------------------
# 2. Graph
# ---------------------------------------------------------------------------

class Graph:
    """Positioned-neuron subgraph with forward (by pre) and reverse (by post) CSR."""

    def __init__(self, body_ids: np.ndarray, types: np.ndarray):
        self.n = len(body_ids)
        self.types = types
        self.order = np.argsort(body_ids, kind="stable")
        self.sorted_ids = body_ids[self.order]

        log("reading edges ...")
        tab = read_feather(EDGES, ["body_pre", "body_post", "weight"])
        self.n_edges_raw = tab.num_rows
        pre = self.to_idx(tab["body_pre"].to_numpy())
        post = self.to_idx(tab["body_post"].to_numpy())
        w = tab["weight"].to_numpy().astype(np.int32)
        del tab

        # total input per positioned neuron counts ALL presynaptic partners
        pm = post >= 0
        self.total_in = np.bincount(post[pm], weights=w[pm], minlength=self.n).astype(np.float64)

        keep = pm & (pre >= 0) & (pre != post)
        pre, post, w = pre[keep].astype(np.int32), post[keep].astype(np.int32), w[keep]
        self.n_edges = len(w)
        log(f"{self.n_edges_raw:,} edges; {self.n_edges:,} between positioned neurons")

        fo = np.lexsort((post, pre))
        self.f_src, self.f_nbr, self.f_w = pre[fo], post[fo], w[fo]
        self.f_ptr = np.concatenate([[0], np.cumsum(np.bincount(pre, minlength=self.n))])
        self.f_nw = self.f_w / np.maximum(self.total_in[self.f_nbr], 1.0)

        ro = np.lexsort((pre, post))
        self.r_src, self.r_nbr, self.r_w = post[ro], pre[ro], w[ro]
        self.r_ptr = np.concatenate([[0], np.cumsum(np.bincount(post, minlength=self.n))])
        self.r_nw = self.r_w / np.maximum(self.total_in[self.r_src], 1.0)
        log("CSR built")

    def to_idx(self, bodies: np.ndarray) -> np.ndarray:
        i = np.searchsorted(self.sorted_ids, bodies)
        i = np.clip(i, 0, self.n - 1)
        hit = self.sorted_ids[i] == bodies
        return np.where(hit, self.order[i], -1)

    # -- connectivity helpers ----------------------------------------------
    def weight_matrix_sum(self, pre_mask: np.ndarray, post_mask: np.ndarray) -> int:
        m = pre_mask[self.f_src] & post_mask[self.f_nbr]
        return int(self.f_w[m].sum())

    def out_weights_to(self, pre_mask: np.ndarray, post_mask: np.ndarray) -> np.ndarray:
        """Per-presynaptic-neuron summed weight onto post_mask set (length n)."""
        m = pre_mask[self.f_src] & post_mask[self.f_nbr]
        return np.bincount(self.f_src[m], weights=self.f_w[m], minlength=self.n)

    def in_weights_from(self, pre_mask: np.ndarray, post_mask: np.ndarray) -> np.ndarray:
        """Per-postsynaptic-neuron summed weight from pre_mask set (length n)."""
        m = pre_mask[self.f_src] & post_mask[self.f_nbr]
        return np.bincount(self.f_nbr[m], weights=self.f_w[m], minlength=self.n)

    # -- path search -------------------------------------------------------
    def best_paths(self, sources, target_mask, reverse=False, max_hops=MAX_HOPS, min_hops=1,
                   k=PATHS_PER_ACTION, per_terminal=MAX_PATHS_PER_TERMINAL, per_prefix=2,
                   per_source=None, rank="bottleneck", fallback=True):
        """Top-k paths (min_hops..max_hops edges) from sources to any target.

        rank="bottleneck": min synapse count along the path (tie-break product).
        rank="product":    product of normalized weights w / total_input(post)
                           (tie-break bottleneck); favors short, dominant routes.
        Exact up to the beam: partial paths are ranked by the best achievable
        completion score (precomputed exact 1- and 2-hop reach bounds).
        Diversity: at most per_terminal paths per terminal neuron, per_prefix
        paths sharing the same start neuron + intermediate cell *types*, one
        path per (start, intermediate types, terminal type), and paths
        balanced across start neurons.
        With reverse=True the search walks upstream and the returned paths
        are flipped so they always read upstream -> downstream.
        """
        if reverse:
            ptr, nbr, ws, nws, srcarr = self.r_ptr, self.r_nbr, self.r_w, self.r_nw, self.r_src
        else:
            ptr, nbr, ws, nws, srcarr = self.f_ptr, self.f_nbr, self.f_w, self.f_nw, self.f_src
        use_prod = rank == "product"

        tg_e = target_mask[nbr]
        nt = ~tg_e
        if use_prod:
            b1 = np.zeros(self.n)
            np.maximum.at(b1, srcarr[tg_e], nws[tg_e])
            b2 = np.zeros(self.n)
            np.maximum.at(b2, srcarr[nt], nws[nt] * b1[nbr[nt]])
        else:
            b1 = np.zeros(self.n, dtype=np.int64)
            np.maximum.at(b1, srcarr[tg_e], ws[tg_e])
            b2 = np.zeros(self.n, dtype=np.int64)
            np.maximum.at(b2, srcarr[nt], np.minimum(ws[nt], b1[nbr[nt]]))
        reach = np.maximum(b1, b2)

        def expand(frontier, bound_arr, keep_nontarget, record_targets):
            """Vectorized expansion of every partial path by one edge."""
            tgt_parts, nxt_parts = [], []
            for fi, (bn, prod, path, pw) in enumerate(frontier):
                a, b = ptr[path[-1]], ptr[path[-1] + 1]
                if a == b:
                    continue
                v, w, nw = nbr[a:b], ws[a:b], nws[a:b]
                ok = np.ones(len(v), dtype=bool)
                for p in path:
                    ok &= v != p
                nb = np.minimum(bn, w)
                npr = prod * nw
                tg = ok & target_mask[v]
                if record_targets and tg.any():
                    j = np.flatnonzero(tg)
                    tgt_parts.append((np.full(len(j), fi), v[j], w[j], nb[j], npr[j]))
                if keep_nontarget:
                    bound = npr * bound_arr[v] if use_prod else np.minimum(nb, bound_arr[v])
                    kp = ok & ~target_mask[v] & (bound > 0)
                    if kp.any():
                        j = np.flatnonzero(kp)
                        nxt_parts.append((np.full(len(j), fi), v[j], w[j], nb[j], npr[j], bound[j]))
            return tgt_parts, nxt_parts

        def cat(parts, n):
            return [np.concatenate([p[i] for p in parts]) for i in range(n)]

        def order(primary_bn, primary_pr):
            return np.lexsort((-primary_bn, -primary_pr)) if use_prod else np.lexsort((-primary_pr, -primary_bn))

        done = []  # (bottleneck, prod, path, weights)
        frontier = [(10 ** 9, 1.0, [int(s)], []) for s in sources]
        for hop in range(1, max_hops + 1):
            bound_arr = reach if hop < max_hops - 1 else b1
            tgt_parts, nxt_parts = expand(frontier, bound_arr, hop < max_hops, hop >= min_hops)
            if tgt_parts:
                fi, v, w, nb, npr = cat(tgt_parts, 5)
                for i in order(nb, npr)[: 100 * k]:
                    bn, prod, path, pw = frontier[fi[i]]
                    done.append((int(nb[i]), float(npr[i]), path + [int(v[i])], pw + [int(w[i])]))
            if not nxt_parts:
                break
            fi, v, w, nb, npr, bound = cat(nxt_parts, 6)
            o = np.argsort(-bound, kind="stable")[:BEAM]
            frontier = [(int(nb[i]), float(npr[i]), frontier[fi[i]][2] + [int(v[i])],
                         frontier[fi[i]][3] + [int(w[i])]) for i in o]

        used_fallback = False
        if not done and fallback:
            used_fallback = True
            done = self._strongest_downstream(sources, ptr, nbr, ws, nws, max_hops)

        done.sort(key=(lambda t: (-t[1], -t[0])) if use_prod else (lambda t: (-t[0], -t[1])))
        n_src = max(1, len(set(p[0] for _, _, p, _ in done)))
        if per_source is None:
            per_source = max(2, math.ceil(k / n_src))
        picked, seen, term_ct, src_ct, pre_ct = [], set(), {}, {}, {}
        for bn, prod, path, pw in done:
            prefix = (path[0],) + tuple(self.types[i] for i in path[1:-1])
            key = prefix + (self.types[path[-1]],)
            if key in seen:
                continue
            t, s = path[-1], path[0]
            if (term_ct.get(t, 0) >= per_terminal or src_ct.get(s, 0) >= per_source
                    or pre_ct.get(prefix, 0) >= per_prefix):
                continue
            seen.add(key)
            term_ct[t] = term_ct.get(t, 0) + 1
            src_ct[s] = src_ct.get(s, 0) + 1
            pre_ct[prefix] = pre_ct.get(prefix, 0) + 1
            picked.append((bn, prod, path, pw))
            if len(picked) >= k:
                break
        if reverse:
            picked = [(bn, prod, path[::-1], pw[::-1]) for bn, prod, path, pw in picked]
        return picked, used_fallback, len(done)

    def _strongest_downstream(self, sources, ptr, nbr, ws, nws, max_hops):
        """Fallback: strongest 2..max_hops-hop paths to anything (beam of 200)."""
        frontier = [(10 ** 9, 1.0, [int(s)], []) for s in sources]
        out = []
        for hop in range(1, max_hops + 1):
            parts = []
            for fi, (bn, prod, path, pw) in enumerate(frontier):
                a, b = ptr[path[-1]], ptr[path[-1] + 1]
                v, w, nw = nbr[a:b], ws[a:b], nws[a:b]
                ok = np.ones(len(v), dtype=bool)
                for p in path:
                    ok &= v != p
                j = np.flatnonzero(ok)
                if len(j):
                    parts.append((np.full(len(j), fi), v[j], w[j], np.minimum(bn, w[j]), prod * nw[j]))
            if not parts:
                break
            fi, v, w, nb, npr = [np.concatenate([p[i] for p in parts]) for i in range(5)]
            o = np.lexsort((-npr, -nb))[:200]
            frontier = [(int(nb[i]), float(npr[i]), frontier[fi[i]][2] + [int(v[i])],
                         frontier[fi[i]][3] + [int(w[i])]) for i in o]
            if hop >= 2:
                out.extend(frontier)
        return out


# ---------------------------------------------------------------------------
# 3. Circuits
# ---------------------------------------------------------------------------

UNIGLOM = re.compile(
    r"^(D|DA[1-4][lm]?|DC[1-4]|DL[1-5][dv]?|DM[1-6]|DP1[lm]|V|VA[1-7][lmdv]?|VC[1-5]|VL[12][ap]?|VM[1-7][dv]?)"
    r"_(adPN|lPN)$")


def main():
    require_inputs(ANNOTATIONS, EDGES)
    OUT.mkdir(parents=True, exist_ok=True)
    df, xyz_raw, all_type_counts = load_neurons()
    pos, centroid, scale, orient_checks = orient(xyz_raw, df)
    N = len(df)

    supers = sorted(df["superclass"].unique())
    assert len(supers) < 256
    sc_idx = df["superclass"].map({s: i for i, s in enumerate(supers)}).values.astype(np.uint8)

    pos.tofile(OUT / "brain_positions.f32")
    sc_idx.tofile(OUT / "brain_superclass.u8")
    bmin, bmax = pos.min(axis=0), pos.max(axis=0)
    meta = {
        "count": int(N),
        "superclasses": supers,
        "superclass_counts": {s: int((sc_idx == i).sum()) for i, s in enumerate(supers)},
        "bounds": {"min": [round(float(v), 5) for v in bmin], "max": [round(float(v), 5) for v in bmax]},
        "bbox_center": [round(float(v), 5) for v in (bmin + bmax) / 2],
        "units_note": (
            f"Soma positions from MaleCNS v1.0 ({VOXEL_NM:.0f} nm voxels), centered on the centroid of all "
            f"{N:,} positioned Traced/Anchor neurons and scaled so the largest extent spans {TARGET_EXTENT} units "
            f"(1 unit ~= {1e-3 * VOXEL_NM / scale:.0f} um). Axes: +x = fly's right, +y = anterior "
            "(brain up, VNC hangs down along -y), +z = dorsal. The dataset's VNC lies posterior to the brain, so "
            "'brain on top' makes anterior point up; a default camera on +z sees a dorsal view."),
        "transform": {
            "centroid_voxels": [round(float(v), 2) for v in centroid],
            "scale_per_voxel": float(scale),
            "three_from_voxel": "x=-(vx-cx)*s, y=-(vz-cz)*s, z=-(vy-cy)*s",
        },
        "orientation_checks": {k: round(v, 4) for k, v in orient_checks.items()},
        "files": {"positions": "brain_positions.f32 (float32 LE, N*3)",
                  "superclass": "brain_superclass.u8 (uint8, N)"},
    }
    (OUT / "brain_meta.json").write_text(json.dumps(meta, indent=1))
    log(f"wrote positions ({N:,} x 3), superclass ({len(supers)} names), meta")

    G = Graph(df["bodyId"].values.astype(np.int64), df["type"].values)

    types = df["type"].values
    cls = df["class"].values
    side = df["somaSide"].values
    sc = df["superclass"].values

    def mask_types(tlist):
        return np.isin(types, list(tlist))

    def idx_of(mask):
        return [int(i) for i in np.flatnonzero(mask)]

    referenced: set[int] = set()

    def ref(ids):
        referenced.update(int(i) for i in ids)
        return [int(i) for i in ids]

    circuits: dict = {"meta": {
        "source": "MaleCNS v1.0 (Janelia/neuPrint), minconf-0.5, significant-only edge table",
        "index_space": "every *_idx / kc / idx / path entry indexes brain_positions.f32 rows",
        "seed": SEED, "kc_sparsity": KC_SPARSITY, "max_hops": MAX_HOPS,
        "path_ranking": "bottleneck (min synapse count along path), tie-break product of w/total_input(post)",
    }}

    # ---- odors / sparse KC code -------------------------------------------
    is_kc = cls == "Kenyon_Cell"
    n_kc = int(is_kc.sum())
    is_alpn = cls == "ALPN"
    glom_types: dict[str, list[str]] = {}
    for t in sorted(set(t for t in types[is_alpn] if t)):
        m = UNIGLOM.match(t)
        if m:
            glom_types.setdefault(m.group(1), []).append(t)
    pn_to_kc = G.out_weights_to(is_alpn, is_kc)
    glom_kc = {g: float(pn_to_kc[mask_types(ts)].sum()) for g, ts in glom_types.items()}
    candidates = sorted(g for g, v in glom_kc.items() if v >= 100)
    rng = np.random.default_rng(SEED)
    draw = [candidates[i] for i in rng.permutation(len(candidates))]
    odor_gloms = {"A": sorted(draw[:N_GLOMERULI]), "B": sorted(draw[N_GLOMERULI:2 * N_GLOMERULI])}
    k_active = int(round(KC_SPARSITY * n_kc))
    NOTES.append(
        f"Odors are synthetic: each is a random draw (seed {SEED}) of {N_GLOMERULI} olfactory glomeruli from "
        f"{len(candidates)} uniglomerular excitatory PN glomeruli (types '<glom>_adPN'/'<glom>_lPN', VP thermo/"
        "hygro glomeruli excluded, >=100 PN->KC synapses). All PN bodies of those types are 'on' with equal rate.")
    NOTES.append(
        f"KC drive = summed real PN->KC synapse counts from the odor's PN bodies. Active KCs = top {KC_SPARSITY:.0%} "
        f"of the {n_kc:,} positioned KCs (k={k_active}) with drive>0: a static k-winners-take-all stand-in for APL "
        "feedback inhibition; no thresholds, dynamics or KC-type differences are modeled.")

    odors, active = {}, {}
    for key in ("A", "B"):
        pn_types = [t for g in odor_gloms[key] for t in glom_types[g]]
        pn_mask = mask_types(pn_types)
        drive = G.in_weights_from(pn_mask, is_kc)
        drive[~is_kc] = 0
        cand = np.flatnonzero(drive > 0)
        top = cand[np.argsort(-drive[cand], kind="stable")][:k_active]
        active[key] = top
        odors[key] = {
            "label": f"Odor {key}",
            "glomeruli": odor_gloms[key],
            "pn_types": pn_types,
            "pn_idx": ref(np.flatnonzero(pn_mask)),
            "kc_idx": ref(top),
            "kc_drive": [int(v) for v in drive[top]],
            "n_kc_with_input": int(len(cand)),
        }
    inter = np.intersect1d(active["A"], active["B"])
    union = np.union1d(active["A"], active["B"])
    overlap = len(inter) / max(1, len(union))
    circuits["odors"] = odors
    circuits["kc_overlap_AB"] = round(overlap, 4)
    circuits["kc_overlap_def"] = "Jaccard |A∩B|/|A∪B| of active KC sets"
    circuits["kc_overlap_count"] = int(len(inter))
    circuits["kc_total"] = n_kc

    # ---- MBONs ----------------------------------------------------------
    mbon_spec = [
        ("MBON11", "γ1pedc>α/β", "approach", "GABA", True),
        ("MBON01", "γ5β'2a", "avoid", "glutamate", True),
        ("MBON12", "γ2α'1", "approach", "ACh", False),
        ("MBON04", "β'2mp_bilateral", "avoid", "glutamate", False),
    ]
    mbons, kc_mbon, mbon_masks = [], {}, {}
    for mtype, comp, role, nt, primary in mbon_spec:
        mm = types == mtype
        if not mm.any():
            NOTES.append(f"{mtype} not found; skipped.")
            continue
        mbon_masks[mtype] = mm
        kc_out = G.out_weights_to(is_kc, mm)
        mbons.append({"type": mtype, "compartment": comp, "role": role, "transmitter": nt, "primary": primary,
                      "idx": ref(np.flatnonzero(mm)),
                      "kc_input_total": int(kc_out.sum()),
                      "n_kc_presynaptic": int((kc_out > 0).sum())})
        kc_mbon[mtype] = {}
        for key in ("A", "B"):
            act = active[key]
            wv = kc_out[act]
            sel = wv > 0
            kc_mbon[mtype][key] = {"kc": ref(act[sel]), "w": [int(v) for v in wv[sel]],
                                   "total": int(wv.sum())}
    circuits["mbons"] = mbons
    circuits["kc_mbon"] = kc_mbon
    NOTES.append(
        "MBON11 (γ1pedc>α/β) and MBON01 (γ5β'2a) are the primary approach/avoid read-outs; MBON12 (γ2α'1, "
        "approach) and MBON04 (β'2mp_bilateral, avoid) are secondary (primary=false). Valence roles follow "
        "Aso et al. 2014 optogenetic valence; the connectome itself carries no valence. kc_mbon lists only active "
        "KCs with >=1 synapse onto the MBON type (w = summed synapses to all bodies of that type).")

    # ---- DANs -----------------------------------------------------------
    dans, dan_report = [], []

    def dan_entry(dtype, signal, mtype):
        dm = types == dtype
        mm = mbon_masks[mtype]
        kc_to_m = G.out_weights_to(is_kc, mm) > 0
        e = {"type": dtype, "signal": signal, "targets": [mtype], "idx": ref(np.flatnonzero(dm)),
             "w_to_mbon": G.weight_matrix_sum(dm, mm),
             "w_to_kc": G.weight_matrix_sum(dm, is_kc),
             "w_to_kc_of_mbon": G.weight_matrix_sum(dm, kc_to_m),
             "w_from_mbon": G.weight_matrix_sum(mm, dm)}
        dan_report.append(f"{dtype:>7} -> {mtype}: direct {e['w_to_mbon']}, onto KCs {e['w_to_kc']} "
                          f"(of which onto {mtype}-presynaptic KCs {e['w_to_kc_of_mbon']}), "
                          f"{mtype} -> {dtype} {e['w_from_mbon']}")
        return e

    def ranked_dans(prefix, mtype):
        dan_types = sorted(set(t for t in types[cls == "DAN"] if t and t.startswith(prefix)))
        mm = mbon_masks[mtype]
        scores = [(G.weight_matrix_sum(types == t, mm), t) for t in dan_types]
        scores.sort(reverse=True)
        return scores

    for mtype in [m["type"] for m in mbons]:
        role = next(m["role"] for m in mbons if m["type"] == mtype)
        if role == "approach":
            if mtype == "MBON11":
                picks = ["PPL101"]
            else:
                picks = [ranked_dans("PPL1", mtype)[0][1]]
            scores = ranked_dans("PPL1", mtype)
            dan_report.append(f"PPL1 types -> {mtype} (direct): " + ", ".join(f"{t} {w}" for w, t in scores[:4]))
            for t in picks:
                dans.append(dan_entry(t, "punishment", mtype))
        else:
            scores = ranked_dans("PAM", mtype)
            dan_report.append(f"PAM types -> {mtype} (direct): " + ", ".join(f"{t} {w}" for w, t in scores[:5]))
            picks = [scores[0][1]]
            if len(scores) > 1 and scores[1][0] >= 0.25 * scores[0][0] and scores[1][0] > 0:
                picks.append(scores[1][1])
            for t in picks:
                dans.append(dan_entry(t, "reward", mtype))
    for d in dans:
        d["primary"] = d["targets"][0] in ("MBON11", "MBON01")
    circuits["dans"] = dans
    NOTES.append(
        "DAN choice: PPL101 (PPL1-γ1pedc) is the canonical punishment DAN for MBON11 and is verified by its "
        "synapses onto MBON11 and onto MBON11-presynaptic KCs (w_to_mbon, w_to_kc_of_mbon). Reward DANs for each "
        "avoid-MBON and the punishment DAN for MBON12 are the DAN types with the most direct synapses onto that "
        "MBON (data-driven; second type kept if >=25% of the first). Dopamine release is volumetric, so direct "
        "DAN->MBON / DAN->KC synapse counts are a proxy for compartment co-innervation, not the plasticity itself.")

    # ---- actions --------------------------------------------------------
    typed = np.array([t is not None for t in types])
    is_mn = (sc == "vnc_motor") & typed       # leg / wing / neck MNs in the VNC
    is_cb_mn = (sc == "cb_motor") & typed     # proboscis / pharynx / neck MNs in the brain
    dng12 = sorted(set(t for t in types if t and t.startswith("DNg12_")))
    fdg = sorted(set(df.loc[df["synonyms"].fillna("").str.contains(r"\bFdg\b"), "type"].dropna()))
    action_spec = [
        ("walk_forward", "Walk forward", ["DNp09"], None, is_mn,
         "DNp09 (P9) drives forward walking (Bidaye et al. 2020)."),
        ("turn_left", "Turn left", ["DNa02"], "L", is_mn,
         "DNa02 steering; left-soma DNa02 activity biases turning to that side (Rayshubskiy et al. 2020)."),
        ("turn_right", "Turn right", ["DNa02"], "R", is_mn, "DNa02 of the right side."),
        ("walk_backward", "Walk backward", ["MDN"], None, is_mn, "MDN, moonwalker DN (Bidaye et al. 2014)."),
        ("takeoff", "Escape takeoff", ["DNp01"], None, is_mn, "DNp01 = giant fiber, escape takeoff."),
        ("groom", "Groom", dng12, None, is_mn,
         "DNg12_* family (all subtypes pooled) as putative head-grooming DNs. Targets restricted to VNC "
         "MNs; in the data their strongest outputs are neck (MNnm*) and front-leg MNs. Allowing brain "
         "cb_motor targets instead yields mostly direct DNg12->neck-MN (CvN4/GNG314) contacts."),
        ("feed", "Feed (proboscis)", fdg or ["GNG588"], None, is_cb_mn,
         "Fdg ('feeding neuron', Flood et al. 2013) = MaleCNS type GNG588 via its 'Shiu 2022: Fdg' synonym. "
         "It is an SEZ interneuron, not a DN; paths target brain (cb_motor) proboscis/pharyngeal motor "
         "neurons (e.g. MN9, rostrum protractor)."),
        ("court", "Court (wing song)", ["pIP10"], None, is_mn, "pIP10, courtship song command DN."),
    ]
    actions = {}
    for key, label, dn_types, want_side, tmask, why in action_spec:
        dm = mask_types(dn_types)
        if want_side:
            dm &= side == want_side
        dn = np.flatnonzero(dm)
        if len(dn) == 0:
            NOTES.append(f"action {key}: no positioned {dn_types} neurons found; empty.")
            actions[key] = {"label": label, "dn_types": dn_types, "dn_idx": [], "paths": []}
            continue
        picked, fb, n_cand = G.best_paths(dn, tmask)
        entry = {
            "label": label, "dn_types": dn_types, "dn_idx": ref(dn),
            "paths": [ref(p) for _, _, p, _ in picked],
            "path_w": [pw for _, _, _, pw in picked],
            "path_bottleneck": [int(bn) for bn, _, _, _ in picked],
            "terminal_types": sorted(set(types[p[-1]] or "?" for _, _, p, _ in picked)),
            "target": "cb_motor" if tmask is is_cb_mn else "vnc_motor",
            "reaches_motor": not fb,
            "note": why,
        }
        if want_side:
            entry["side"] = want_side
        if fb:
            NOTES.append(f"action {key}: no positioned path to a motor neuron within {MAX_HOPS} hops; "
                         "paths are the strongest 2-3 hop downstream chains instead.")
        actions[key] = entry
    circuits["actions"] = actions
    NOTES.append(
        f"Action paths: up to {PATHS_PER_ACTION} paths of <= {MAX_HOPS} synapses from the DN(s) to a typed motor "
        f"neuron (vnc_motor for every action except feed, which targets cb_motor), through positioned neurons "
        f"only. Ranked by bottleneck (weakest synapse count on the path). Diversity caps: <= {MAX_PATHS_PER_TERMINAL} "
        "paths per terminal MN, <= 2 per identical start+intermediate-type chain, balanced across DN bodies. Beam "
        f"search (width {BEAM}) with exact 1-/2-hop reach bounds. Product-of-normalized-weights ranking was tried "
        "and rejected: it favors 2-10 synapse contacts onto small MNs. These are strong anatomical routes, not "
        "simulated activity; electrical synapses (e.g. GF->TTMn gap junctions) are absent from the EM connectome.")
    NOTES.append("DNa02 L/R split uses somaSide (rootSide is empty for these DNs).")
    for key in ("groom", "feed"):
        NOTES.append(f"{key}: {actions[key]['note']}")

    # ---- looming --------------------------------------------------------
    lplc2 = types == "LPLC2"
    lc4 = types == "LC4"
    gf = types == "DNp01"
    loom_paths, loom_w, loom_bn = [], [], []
    loom_direct = {}
    loom_kind = []
    gf_idx = np.flatnonzero(gf)
    half = PATHS_PER_LOOM_TYPE // 2
    for name, m in (("LPLC2", lplc2), ("LC4", lc4)):
        for kind, lo, hi in (("direct", 1, 1), ("indirect", 2, MAX_HOPS)):
            picked, fb, _ = G.best_paths(gf_idx, m, reverse=True, min_hops=lo, max_hops=hi, k=half,
                                         per_terminal=1, per_prefix=1, per_source=math.ceil(half / len(gf_idx)),
                                         fallback=False)
            for bn, _, p, pw in picked:
                loom_paths.append(ref(p))
                loom_w.append(pw)
                loom_bn.append(int(bn))
                loom_kind.append(f"{name}:{kind}")
        loom_direct[name] = G.weight_matrix_sum(m, gf)
    circuits["looming"] = {
        "lplc2_idx": ref(np.flatnonzero(lplc2)),
        "lc4_idx": ref(np.flatnonzero(lc4)),
        "gf_idx": ref(np.flatnonzero(gf)),
        "paths": loom_paths, "path_w": loom_w, "path_bottleneck": loom_bn, "path_kind": loom_kind,
        "direct_w_to_DNp01": loom_direct,
    }
    NOTES.append(
        f"Looming: all LPLC2 and LC4 (both hemispheres). Paths into DNp01: per type, the {half} strongest direct "
        f"VPN->GF synapses (one per GF side) plus the {half} strongest 2-{MAX_HOPS} hop routes (distinct "
        "intermediate types), ranked by bottleneck. Single-VPN direct contacts are modest (~60-90 synapses) but "
        "the populations converge heavily on the GF (direct_w_to_DNp01 = summed over all bodies).")

    # ---- group members dropped for lack of a position ---------------------
    named = set(t for o in odors.values() for t in o["pn_types"])
    named |= {m["type"] for m in mbons} | {d["type"] for d in dans} | {"LPLC2", "LC4", "DNp01"}
    named |= set(t for a in actions.values() for t in a["dn_types"])
    pos_counts = pd.Series(types).value_counts().to_dict()
    dropped = {t: (pos_counts.get(t, 0), all_type_counts.get(t, 0)) for t in sorted(named)
               if pos_counts.get(t, 0) < all_type_counts.get(t, 0)}
    if dropped:
        NOTES.append("Bodies without a soma position were dropped from these named groups (positioned/total): "
                     + ", ".join(f"{t} {a}/{b}" for t, (a, b) in dropped.items()) + ".")
    NOTES.append(f"KC pool = the {n_kc:,} positioned Kenyon cells (of {all_type_counts['__KC_TOTAL__']:,} traced); "
                 "KCs without a soma position are excluded from the k-WTA.")
    NOTES.append("Orientation (brain_meta.units_note): +x = fly's right, +y = anterior (brain up, VNC hangs down), "
                 "+z = dorsal. 'Anterior facing +z' is impossible together with 'VNC below' because the VNC lies "
                 "posterior to the brain in this volume; a camera on +z sees the CNS from above (dorsal view).")

    # ---- labels for every referenced neuron ------------------------------
    bid = df["bodyId"].values
    circuits["neurons"] = {
        str(i): [int(bid[i]), types[i] or "", side[i] or "", sc[i]] for i in sorted(referenced)}
    circuits["neurons_fields"] = ["bodyId", "type", "somaSide", "superclass"]
    circuits["notes"] = NOTES

    # ---- checks + write ---------------------------------------------------
    assert all(0 <= i < N for i in referenced), "index out of range"
    for k, o in odors.items():
        assert o["pn_idx"] and o["kc_idx"], f"empty odor {k}"
    empty = [k for k, a in actions.items() if not a["paths"]]
    if empty:
        NOTES.append(f"actions with no paths: {empty}")
    txt = json.dumps(circuits, ensure_ascii=False, separators=(",", ":"))
    (OUT / "circuits.json").write_text(txt, encoding="utf-8")

    # ---- summary -----------------------------------------------------------
    print("\n==== SUMMARY ====")
    print(f"positioned neurons: {N:,}  (superclasses: {len(supers)})")
    print(f"edges: {G.n_edges_raw:,} total, {G.n_edges:,} between positioned neurons")
    print(f"KCs: {n_kc:,}; active per odor k={k_active}")
    for k, o in odors.items():
        print(f"odor {k}: glomeruli {o['glomeruli']}  PN types {len(o['pn_types'])}  PNs {len(o['pn_idx'])}  "
              f"KCs w/ input {o['n_kc_with_input']}  active {len(o['kc_idx'])}  "
              f"drive range {min(o['kc_drive'])}-{max(o['kc_drive'])}")
    print(f"KC overlap A/B: {len(inter)} shared, Jaccard {overlap:.3f}")
    for m in mbons:
        a, b = kc_mbon[m['type']]["A"], kc_mbon[m['type']]["B"]
        print(f"{m['type']} ({m['compartment']}, {m['role']}): {len(m['idx'])} bodies, KC input {m['kc_input_total']} "
              f"from {m['n_kc_presynaptic']} KCs | odor A: {len(a['kc'])} KCs w={a['total']} | "
              f"odor B: {len(b['kc'])} KCs w={b['total']}")
    print("DAN -> MBON:")
    for line in dan_report:
        print("  " + line)
    print("DANs chosen: " + ", ".join(f"{d['type']}({d['signal']}->{d['targets'][0]})" for d in dans))
    print("Actions:")
    for k, a in actions.items():
        print(f"  {k:14s} DNs {a['dn_types'][:3]}{'...' if len(a['dn_types']) > 3 else ''} n={len(a['dn_idx'])} "
              f"paths={len(a['paths'])} bottlenecks={a.get('path_bottleneck')} "
              f"hops={[len(p) - 1 for p in a['paths']]} reaches_MN={a.get('reaches_motor')} "
              f"terminals={a.get('terminal_types')}")
    lm = circuits["looming"]
    print(f"Looming: LPLC2 {len(lm['lplc2_idx'])}, LC4 {len(lm['lc4_idx'])}, paths {len(lm['paths'])} "
          f"bottlenecks {lm['path_bottleneck']} direct->GF {lm['direct_w_to_DNp01']}")
    print(f"referenced neurons: {len(referenced)}")
    print("files:")
    for f in ("brain_positions.f32", "brain_superclass.u8", "brain_meta.json", "circuits.json"):
        print(f"  data/{f:22s} {(OUT / f).stat().st_size / 1e6:8.3f} MB")
    assert (OUT / "brain_positions.f32").stat().st_size == N * 12
    assert (OUT / "brain_superclass.u8").stat().st_size == N
    log("done")


if __name__ == "__main__":
    sys.exit(main())
