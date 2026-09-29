#!/usr/bin/env python3
"""Compound eye + visual pathway for the arena, from the local MaleCNS v1.0 data.

Run after build_circuits.py; takes the same MaleCNS input dir option
(--malecns-dir PATH, else $MALECNS_DIR, else ./male-cns-data at the repo root):

    python3 arena/build_vision.py [--malecns-dir PATH]

Writes into ./data/:
    vision.json     eye columns (side, hex coords, viewing direction, which
                    positioned column neurons to light), visual projection
                    targets (LC*/LPLC* neurons: brain index, type, side,
                    receptive-field centre), notes
    vision_w.bin    sparse column -> target weights, little-endian:
                    int32 col[n], int32 tgt[n], float32 w[n]  (n in vision.json)

What is real: which neurons exist, their eye-column (hex) assignment, their
positions, their synapses and the predicted neurotransmitter of every
presynaptic neuron. What is modelled: the viewing direction of each column
(fitted from lamina geometry, then stretched to a typical fly eye field) and
the linear, signed two-hop propagation (no dynamics, so no true motion
computation).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import scipy.sparse as sp

from build_circuits import ANNOTATIONS, EDGES, OUT, SRC, load_neurons, log, orient, read_feather, require_inputs

DRIVE_ACTIONS = ("walk_forward", "veer_left", "veer_right", "walk_backward", "takeoff")
DRIVE_CIRCUIT = {"walk_forward": "walk_forward", "veer_left": "turn_left", "veer_right": "turn_right", "walk_backward": "walk_backward", "takeoff": "takeoff"}

NT_FILE = SRC / "body-neurotransmitters-male-cns-v1.0.feather"
LAMINA_TYPES = ("L1", "L2", "L3", "L5")
TARGET_RE = r"^(LC|LPLC)\d"               # visual projection neurons we read out
TOP_K = 64                                # strongest column inputs kept per target
AZ_FRONT, AZ_REAR = -15.0, 165.0          # typical field of one eye (deg, + = own side)
EL_LOW, EL_HIGH = -60.0, 70.0
NT_SIGN = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0, "histamine": -1.0}
NOTES: list[str] = []


def main():
    require_inputs(ANNOTATIONS, EDGES, NT_FILE)
    # ---- positioned neurons, same order as brain_positions.f32
    pdf, xyz, _ = load_neurons()
    pos3, *_ = orient(xyz, pdf)
    pidx = pd.Series(np.arange(len(pdf)), index=pdf["bodyId"].values)

    # ---- every traced neuron (positioned or not) with type, side, eye column
    tab = read_feather(ANNOTATIONS, ["bodyId", "type", "superclass", "status", "somaSide", "assignedOlHex1", "assignedOlHex2"])
    tab = tab.filter(pc.is_in(tab["status"], value_set=pa.array(["Traced", "Anchor"])))
    a = tab.to_pandas()
    a["type"] = a["type"].astype(object)
    a["superclass"] = a["superclass"].astype(object).fillna("unclassified")
    a["idx"] = pidx.reindex(a["bodyId"].values).fillna(-1).astype(int).values

    # ---- neurotransmitter sign of every body
    nt = read_feather(NT_FILE, ["body", "consensus_nt"]).to_pandas()
    sign = pd.Series(nt["consensus_nt"].map(NT_SIGN).fillna(0.0).values, index=nt["body"].values)
    a["sign"] = sign.reindex(a["bodyId"].values).fillna(0.0).values
    log(f"NT signs: {int((a.sign > 0).sum()):,} excitatory, {int((a.sign < 0).sum()):,} inhibitory, {int((a.sign == 0).sum()):,} neutral/unknown")

    # ---- eye columns
    hx = a[a["assignedOlHex1"].notna() & a["somaSide"].isin(["L", "R"])].copy()
    hx["h1"] = hx["assignedOlHex1"].astype(int)
    hx["h2"] = hx["assignedOlHex2"].astype(int)
    cols = hx[["somaSide", "h1", "h2"]].drop_duplicates().sort_values(["somaSide", "h1", "h2"]).reset_index(drop=True)
    cols["cid"] = np.arange(len(cols))
    hx = hx.merge(cols, on=["somaSide", "h1", "h2"])
    log(f"{len(hx):,} column neurons in {len(cols):,} eye columns "
        f"(L {int((cols.somaSide == 'L').sum())}, R {int((cols.somaSide == 'R').sum())}), types: {sorted(hx.type.unique())}")

    # ---- viewing directions from lamina geometry
    # Hex coordinates are mirror-consistent between the eyes (checked: corr >= 0.96 on shared columns), so both
    # eyes' lamina neurons are pooled in a left-eye frame (right eye mirrored across the midline) for one fit.
    lam = hx[hx.type.isin(LAMINA_TYPES) & (hx.idx >= 0)]
    P = pos3[lam.idx.values].astype(float)
    P[lam.somaSide.values == "R", 0] *= -1                                   # mirror right eye onto the left
    H = lam[["h1", "h2"]].values.astype(float)
    design = lambda h: np.c_[np.ones(len(h)), h[:, 0], h[:, 1], h[:, 0] ** 2, h[:, 0] * h[:, 1], h[:, 1] ** 2]
    coef, *_ = np.linalg.lstsq(design(H), P, rcond=None)
    rms = float(np.sqrt(((design(H) @ coef - P) ** 2).sum(1).mean()))
    Pc = design(cols[["h1", "h2"]].values.astype(float)) @ coef              # eye-surface point per column (left frame)
    A = np.c_[2 * Pc, np.ones(len(Pc))]
    sol, *_ = np.linalg.lstsq(A, (Pc ** 2).sum(1), rcond=None)
    D = Pc - sol[:3]
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    # Three.js brain frame: +x fly's right, +y anterior, +z dorsal -> robot frame (left, up, forward); left eye frame
    left, up, fwd = -D[:, 0], D[:, 2], D[:, 1]
    raw_az = np.degrees(np.arctan2(left, fwd)); raw_el = np.degrees(np.arcsin(np.clip(up, -1, 1)))
    log(f"joint lamina fit on {len(lam)} neurons (both eyes), rms {rms:.3f} units; raw az {np.percentile(raw_az, 2):.0f}..{np.percentile(raw_az, 98):.0f}, "
        f"raw el {np.percentile(raw_el, 2):.0f}..{np.percentile(raw_el, 98):.0f}")
    def stretch(v, lo, hi):
        a0, a1 = np.percentile(v, 2), np.percentile(v, 98)
        return np.clip(lo + (v - a0) / max(a1 - a0, 1e-6) * (hi - lo), lo, hi)
    s = np.where(cols.somaSide.values == "L", 1.0, -1.0)                      # + = the eye's own side
    az = s * stretch(raw_az, AZ_FRONT, AZ_REAR)
    el = stretch(raw_el, EL_LOW, EL_HIGH)
    cols["az"] = az; cols["el"] = el
    NOTES.append(
        "Column viewing directions: one quadratic map from hex coordinates to lamina soma positions was fitted on the positioned "
        "L1/L2/L3/L5 neurons of both eyes (right eye mirrored; hex coordinates are mirror-consistent); directions point outward "
        "from a sphere fitted to that sheet, so the two eyes are exact mirror images. Their spread was then "
        f"stretched to a typical fly eye field (azimuth {AZ_FRONT:g}..{AZ_REAR:g} deg per eye, elevation {EL_LOW:g}..{EL_HIGH:g} deg). "
        "Ordering comes from the data; absolute angles are approximate.")

    # ---- targets (visual projection neurons we read out) and intermediates for the second hop
    tgt = a[a.type.astype(str).str.match(TARGET_RE) & (a.idx >= 0) & a.somaSide.isin(["L", "R"])].reset_index(drop=True)
    tgt["tid"] = np.arange(len(tgt))
    hx_bodies = set(hx.bodyId.values)
    mids = a[(a.superclass == "ol_intrinsic") & ~a.bodyId.isin(hx_bodies)].reset_index(drop=True)
    mids["mid"] = np.arange(len(mids))
    log(f"targets: {len(tgt):,} LC/LPLC neurons in {tgt.type.nunique()} types; second-hop intermediates: {len(mids):,} optic-lobe neurons")

    # ---- edges
    e = read_feather(EDGES, ["body_pre", "body_post", "weight"]).to_pandas()
    total_in = e.groupby("body_post")["weight"].sum()
    col_of = pd.Series(hx.cid.values, index=hx.bodyId.values)
    sgn = pd.Series(a.sign.values, index=a.bodyId.values)
    t_of = pd.Series(tgt.tid.values, index=tgt.bodyId.values)
    m_of = pd.Series(mids.mid.values, index=mids.bodyId.values)

    def block(pre_map, post_map, rows, ncols):
        m = e.body_pre.isin(pre_map.index) & e.body_post.isin(post_map.index)
        sub = e[m]
        w = sub.weight.values * sgn.reindex(sub.body_pre.values).fillna(0).values
        w = w / total_in.reindex(sub.body_post.values).fillna(1).values        # fraction of the target's input
        M = sp.coo_matrix((w, (pre_map.reindex(sub.body_pre.values).values, post_map.reindex(sub.body_post.values).values)),
                          shape=(rows, ncols)).tocsr()
        M.sum_duplicates()
        return M, int(m.sum())

    W1, n1 = block(col_of, t_of, len(cols), len(tgt))          # column neurons -> LC/LPLC directly
    A1, n2 = block(col_of, m_of, len(cols), len(mids))         # column neurons -> T4/T5/TmY/Li/...
    A2, n3 = block(m_of, t_of, len(mids), len(tgt))            # intermediates -> LC/LPLC
    W2 = (A1 @ A2).tocsr()
    W = (W1 + W2).tocoo()
    log(f"edges used: direct {n1:,}, col->mid {n2:,}, mid->target {n3:,}; combined nnz {W.nnz:,}")

    # prune: keep the strongest TOP_K column inputs per target, then normalise each target's input to unit L1
    df = pd.DataFrame({"c": W.row, "t": W.col, "w": W.data})
    df = df[df.w != 0]
    df["aw"] = df.w.abs()
    df = df.sort_values(["t", "aw"], ascending=[True, False]).groupby("t").head(TOP_K)
    df["w"] = df.w / df.groupby("t").aw.transform("sum")
    covered = df.t.nunique()
    log(f"kept {len(df):,} weights; {covered:,}/{len(tgt):,} targets receive column input "
        f"({(df.w > 0).mean() * 100:.0f}% of kept weights excitatory)")

    # receptive-field centres from positive inputs
    pos = df[df.w > 0]
    rf = pos.assign(az=cols.az.values[pos.c], el=cols.el.values[pos.c]).groupby("t").apply(
        lambda g: pd.Series({"az": np.average(g.az, weights=g.w), "el": np.average(g.el, weights=g.w)}), include_groups=False)
    tgt["rf_az"] = rf.az.reindex(tgt.tid).fillna(0).values
    tgt["rf_el"] = rf.el.reindex(tgt.tid).fillna(0).values
    tgt["has_input"] = tgt.tid.isin(df.t.unique())
    for ty in ("LC4", "LPLC2", "LC11", "LC16", "LPLC1"):
        g = tgt[(tgt.type == ty) & tgt.has_input]
        if len(g):
            log(f"  {ty}: {len(g)} neurons with input; RF az L {g[g.somaSide == 'L'].rf_az.mean():.0f}, R {g[g.somaSide == 'R'].rf_az.mean():.0f}")

    # which positioned column neurons light up per column (medulla/lobula types are positioned)
    light = hx[hx.idx >= 0].groupby("cid").idx.apply(lambda s: [int(v) for v in s.values[:8]])
    light = light.reindex(cols.cid).apply(lambda v: v if isinstance(v, list) else []).tolist()

    # ---- innate visuomotor drive: LC/LPLC neurons -> the command neurons behind each motor program
    circuits = json.loads((OUT / "circuits.json").read_text())
    body_of_idx = pdf["bodyId"].values
    dn_bodies = {a: [int(body_of_idx[i]) for i in circuits["actions"][DRIVE_CIRCUIT[a]]["dn_idx"]] for a in DRIVE_ACTIONS}
    dn_col = pd.Series({b: k for k, a in enumerate(DRIVE_ACTIONS) for b in dn_bodies[a]})
    all_ids = a.bodyId.values
    all_of = pd.Series(np.arange(len(all_ids)), index=all_ids)
    # direct: target -> DN
    D1, n_d1 = block(t_of, dn_col, len(tgt), len(DRIVE_ACTIONS))
    # one intermediate (any neuron): target -> mid -> DN
    T2, n_t2 = block(t_of, all_of, len(tgt), len(all_ids))
    M2, n_m2 = block(all_of, dn_col, len(all_ids), len(DRIVE_ACTIONS))
    drive = (D1 + T2 @ M2).toarray()                                      # targets x actions, signed
    full = np.maximum(drive, 0).sum(axis=0)                               # drive if every LC/LPLC fired at once
    drive = drive / np.maximum(full, 1e-9)                                # full-field activation -> 1.0 per action
    log(f"visuomotor: direct {n_d1:,} synapses, target->mid {n_t2:,}, mid->DN {n_m2:,}")
    for k, act in enumerate(DRIVE_ACTIONS):
        per_type = pd.Series(np.maximum(drive[:, k], 0)).groupby(tgt.type.values).sum().sort_values(ascending=False)
        log(f"  {act:13} ({'/'.join(sorted(set(circuits['actions'][DRIVE_CIRCUIT[act]]['dn_types'])))}): strongest "
            + ", ".join(f"{t} {v:.2f}" for t, v in per_type.head(4).items()))
    NOTES.append("Visuomotor drive: real signed synapses from each LC/LPLC neuron to the command neurons of five motor programs "
                 "(DNp09 walk, DNa02 left/right veer, MDN back, DNp01 boosters), direct or via one intermediate neuron, each "
                 "column scaled so that all LC/LPLC neurons firing together gives 1. The plastic gains that sit on these "
                 "pathways at runtime are a modelling choice, not measured biology.")

    type_names = sorted(tgt.type.unique())
    type_ix = {t: i for i, t in enumerate(type_names)}
    out = {
        "columns": {
            "side": cols.somaSide.tolist(),
            "hex": cols[["h1", "h2"]].values.astype(int).tolist(),
            "az": np.round(cols.az.values, 2).tolist(),
            "el": np.round(cols.el.values, 2).tolist(),
            "light": light,
        },
        "targets": {
            "idx": tgt.idx.astype(int).tolist(),
            "type": [type_ix[t] for t in tgt.type],
            "typeNames": type_names,
            "side": tgt.somaSide.tolist(),
            "rf_az": np.round(tgt.rf_az.values, 1).tolist(),
            "rf_el": np.round(tgt.rf_el.values, 1).tolist(),
            "has_input": tgt.has_input.astype(bool).tolist(),
        },
        "n_weights": int(len(df)),
        "drive": {"actions": list(DRIVE_ACTIONS), "w": np.round(drive, 5).tolist()},
        "notes": NOTES + [
            f"Targets are the {len(tgt):,} positioned LC*/LPLC* visual projection neurons ({len(type_names)} types).",
            "Propagation: each eye column's signal drives its real column neurons (L/Mi/Tm/T1/C types with hex assignments); "
            "their real synapses reach the targets directly and via one optic-lobe intermediate (T4/T5, TmY, Li, ...). "
            "Each synapse is signed by the presynaptic neuron's consensus neurotransmitter (ACh +, GABA/Glu/histamine -, others 0) "
            "and scaled by its share of the postsynaptic neuron's total input.",
            f"Only the {TOP_K} strongest column inputs per target are kept; each target's kept inputs are L1-normalised.",
            "This is a static linear read-through of real wiring, not a dynamical simulation: it cannot reproduce motion "
            "direction selectivity; change over time is supplied as an input feature instead.",
        ],
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "vision.json").write_text(json.dumps(out, separators=(",", ":")))
    with open(OUT / "vision_w.bin", "wb") as fh:
        fh.write(df.c.values.astype("<i4").tobytes())
        fh.write(df.t.values.astype("<i4").tobytes())
        fh.write(df.w.values.astype("<f4").tobytes())
    log(f"wrote vision.json ({(OUT / 'vision.json').stat().st_size / 1e6:.2f} MB) and vision_w.bin ({(OUT / 'vision_w.bin').stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
