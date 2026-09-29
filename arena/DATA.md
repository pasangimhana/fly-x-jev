# Arena data: what the build scripts produce

`data/` is committed prebuilt, so running the demo does not need any of this. You only need the MaleCNS download to rebuild `data/`.

Two scripts, run in order from the repo root (need `pyarrow`, `pandas`, `numpy`; `build_vision.py` also needs `scipy`):

```
python3 arena/build_circuits.py   # brain_positions.f32, brain_superclass.u8, brain_meta.json, circuits.json
python3 arena/build_vision.py     # vision.json, vision_w.bin (reads circuits.json)
```

All neuron references in the output are row indices into `brain_positions.f32`.

## Inputs

Input directory, in order of precedence: the `--malecns-dir PATH` flag (both scripts accept it), the `MALECNS_DIR` environment variable, else `male-cns-data/` at the repo root (gitignored).

The files are from the public MaleCNS v1.0 release (FlyEM / HHMI Janelia, University of Cambridge, MRC LMB, Google Research), licensed CC-BY 4.0. Download page: https://male-cns.janelia.org/download/. The "significant-only" edge table is not linked from that page but is in the same public bucket (`gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`).

| File | Bytes | Used by |
|---|---|---|
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 14,483,314 | both: type, class, superclass, status, soma location, side, synonyms, eye-column hex assignment |
| `connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather` | 502,169,298 | both: `body_pre`, `body_post`, `weight` (synapse count) |
| `body-neurotransmitters-male-cns-v1.0.feather` | 43,282,834 | `build_vision.py` only: `body`, `consensus_nt` |

```
mkdir -p male-cns-data && cd male-cns-data
BASE=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
curl -LO $BASE/body-annotations-male-cns-v1.0-minconf-0.5.feather
curl -LO $BASE/connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather
curl -LO $BASE/body-neurotransmitters-male-cns-v1.0.feather
```

(or `gsutil cp gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/<file> .`)

## `build_circuits.py` outputs (`data/`)

| File | Size | Content |
|---|---|---|
| `brain_positions.f32` | 1,694,496 B | float32 LE, N x 3 soma positions in Three.js coords (N = 141,208) |
| `brain_superclass.u8` | 141,208 B | uint8 per neuron, index into `brain_meta.superclasses` (21 names) |
| `brain_meta.json` | 2,015 B | count, superclasses + counts, bounds, bbox_center, units_note, voxel transform, orientation_checks, files |
| `circuits.json` | 77,739 B | odor, mushroom-body, DAN, action and looming circuits (schema below) |

Build steps: keep Traced/Anchor neurons with a soma coordinate; recentre and scale so the largest extent is 2.0 units (1 unit ~ 498 um; +x fly right, +y anterior, +z dorsal); read the edge table, drop edges to unpositioned neurons and self-edges; pick circuits below; write files and assert file sizes (N x 12 and N bytes).

## `circuits.json` schema

- `meta`: source, index_space, seed (7), kc_sparsity (0.05), max_hops (3), path_ranking (bottleneck = min synapse count on the path, tie-break product of w / total input of post).
- `odors.{A,B}`: label, glomeruli, pn_types, pn_idx (PN bodies), kc_idx (active KCs), kc_drive (PN->KC synapses onto each active KC), n_kc_with_input.
- `kc_overlap_AB`, `kc_overlap_def`, `kc_overlap_count`, `kc_total`: Jaccard overlap of the A and B active KC sets, shared count, size of KC pool.
- `mbons[]`: type, compartment, role, transmitter, primary, idx, kc_input_total (all KC->MBON synapses), n_kc_presynaptic.
- `kc_mbon.{MBON}.{A,B}`: kc (active KCs with >=1 synapse onto the type), w (summed synapses per KC), total.
- `dans[]`: type, signal (punishment/reward), targets, idx, primary, plus measured synapse sums: w_to_mbon, w_to_kc, w_to_kc_of_mbon (KCs presynaptic to that MBON), w_from_mbon.
- `actions.{name}`: label, dn_types, dn_idx, paths (index chains, DN first), path_w (synapses per hop), path_bottleneck, terminal_types, target (vnc_motor or cb_motor), reaches_motor, note, side (turn_left/turn_right only).
- `looming`: lplc2_idx, lc4_idx, gf_idx, paths (VPN first, GF last), path_w, path_bottleneck, path_kind, direct_w_to_DNp01.
- `neurons`: index (string) -> [bodyId, type, somaSide, superclass] for every referenced index (1,040 entries); `neurons_fields` names the columns.
- `notes`: 13 strings, restated under Approximations.

## Concrete choices

**Odors (synthetic).** Seed 7 random draw of 6 glomeruli each from 48 candidates; every PN body of those types is on.
- A: DL1, DM5, DP1l, VA4, VL2a, VM5v. 22 PN bodies, 2,052 KCs get input, drive 43-133 synapses.
- B: D, DA4l, DC1, DL2v, DM4, VC4. 26 PN bodies, 2,149 KCs get input, drive 49-111 synapses.
- KC code: top 5% of 4,054 positioned KCs, k = 203 per odor. A and B share 9 KCs (Jaccard 0.0227).

**MBONs.** MBON11 and MBON01 are primary; roles from Aso et al. 2014, not from the connectome.

| Type | Compartment | Role | Transmitter | Bodies | KC input | Active KCs A / B (summed w) |
|---|---|---|---|---|---|---|
| MBON11 | γ1pedc>α/β | approach (primary) | GABA | 2 | 41,383 from 3,614 KCs | 195 (3,163) / 189 (2,792) |
| MBON01 | γ5β'2a | avoid (primary) | glutamate | 2 | 40,831 from 2,078 KCs | 155 (3,424) / 141 (3,121) |
| MBON12 | γ2α'1 | approach (secondary) | ACh | 4 | 22,554 from 2,173 KCs | 155 (1,527) / 152 (1,406) |
| MBON04 | β'2mp_bilateral | avoid (secondary) | glutamate | 2 | 6,785 from 1,585 KCs | 110 (326) / 107 (314) |

**DANs** (synapse sums over all bodies of the type; bodies in brackets).

| Type | Signal | Target | to MBON | to all KCs | to KCs of MBON | MBON back | Primary |
|---|---|---|---|---|---|---|---|
| PPL101 [2] | punishment | MBON11 | 2,311 | 11,445 | 11,413 | 205 | yes |
| PAM01 [44] | reward | MBON01 | 1,597 | 24,006 | 23,988 | 128 | yes |
| PAM02 [17] | reward | MBON01 | 550 | 4,914 | 4,262 | 16 | yes |
| PPL103 [2] | punishment | MBON12 | 802 | 20,483 | 19,419 | 53 | no |
| PAM05 [20] | reward | MBON04 | 727 | 11,024 | 10,965 | 143 | no |
| PAM06 [28] | reward | MBON04 | 677 | 18,345 | 18,038 | 60 | no |

**Actions** (8 paths each, <= 3 synapses, DN to motor neuron; the app's Jev choice list (`js/jev.js`) uses six of them: walk_forward, turn_left, turn_right, walk_backward, takeoff and feed; `groom` and `court` are built but unused).
- walk_forward: DNp09 (2 bodies) -> ADNM1 MN, ADNM2 MN, Acc. ti flexor MN, MNnm11, Pleural remotor/abductor MN, Sternal anterior rotator MN, Ti extensor MN, Tr extensor MN. DNp09 (P9) drives forward walking (Bidaye et al. 2020).
- turn_left: DNa02 left soma (1) -> MNml29, Pleural remotor/abductor MN, Sternal anterior/posterior rotator MN. Left DNa02 activity biases turning to that side (Rayshubskiy et al. 2020).
- turn_right: DNa02 right soma (1) -> ADNM2 MN, FNM2, Pleural remotor/abductor MN, Sternal anterior/posterior rotator MN. DNa02 of the right side.
- walk_backward: MDN (4) -> Ti extensor MN, Ti flexor MN. Moonwalker DN (Bidaye et al. 2014).
- takeoff: DNp01 (2) -> DVMn 1a-c, Pleural remotor/abductor MN, TTMn, Tr flexor MN, i2 MN, iii1 MN, ps1 MN. Giant fiber, escape takeoff.
- groom: DNg12_a to _h (41 bodies pooled) -> MNnm03, MNnm11, MNnm13, Sternal anterior rotator MN, Tergopleural/Pleural promotor MN. Putative head-grooming DNs.
- feed: GNG588 = Fdg (2 bodies; an SEZ interneuron, not a DN) -> GNG314, MN2Da, MN2Db, MN2V, MN4b, MN9 (target cb_motor). Fdg 'feeding neuron' (Flood et al. 2013).
- court: pIP10 (2) -> hg1 MN, i2 MN, tpn MN. Courtship song command DN.

**Looming.** LPLC2 (185 bodies) and LC4 (126 bodies) converge on DNp01 (giant fiber, 2 bodies). 8 paths, 4 per type: 2 direct VPN->GF and 2 indirect 3-hop. Bottlenecks: LPLC2 direct 64, 59; LPLC2 indirect 98, 80; LC4 direct 86, 67; LC4 indirect 172, 154. Summed direct synapses onto DNp01: LPLC2 4,862; LC4 6,362.

## Approximations (the 13 `notes`)

1. Positions are soma coordinates (somaLocation, with tosomaLocation as fallback for 976 cells). 24,525 of 165,733 Traced/Anchor neurons have neither, mostly sensory neurons whose somata lie outside the CNS, and are excluded from every array and every path.
2. Odors are synthetic: a random draw (seed 7) of 6 glomeruli from 48 uniglomerular excitatory PN glomeruli (types `<glom>_adPN` / `_lPN`; VP thermo/hygro glomeruli excluded; at least 100 PN->KC synapses). All PN bodies of those types are on at equal rate.
3. KC drive is the summed real PN->KC synapse count from the odor's PN bodies. Active KCs are the top 5% of the 4,054 positioned KCs (k = 203) with drive > 0. This is a static k-winners-take-all stand-in for APL feedback inhibition; no thresholds, dynamics or KC-type differences are modeled.
4. MBON11 and MBON01 are the primary approach/avoid read-outs; MBON12 (approach) and MBON04 (avoid) are secondary. Valence roles follow Aso et al. 2014 optogenetic valence; the connectome carries no valence. `kc_mbon` lists only active KCs with at least 1 synapse onto the MBON type (w sums synapses to all bodies of the type).
5. DAN choice: PPL101 is the canonical punishment DAN for MBON11, checked by its synapses onto MBON11 and onto MBON11-presynaptic KCs. Reward DANs for each avoid-MBON, and the punishment DAN for MBON12, are the DAN types with the most direct synapses onto that MBON (data-driven; a second type is kept if it has at least 25% of the first). Dopamine release is volumetric, so direct DAN->MBON and DAN->KC counts are a proxy for compartment co-innervation, not the plasticity itself.
6. Action paths are up to 8 paths of at most 3 synapses from the DN(s) to a typed motor neuron (vnc_motor, except feed which targets cb_motor), through positioned neurons only. Ranked by bottleneck. Caps: at most 2 paths per terminal MN, at most 2 per identical start-plus-intermediate-type chain, balanced across DN bodies. Beam search (width 4000) with exact 1- and 2-hop reach bounds. Ranking by product of normalized weights was tried and rejected because it favors 2-10 synapse contacts onto small MNs. These are strong anatomical routes, not simulated activity; electrical synapses (e.g. GF->TTMn gap junctions) are absent from the EM connectome.
7. The DNa02 left/right split uses somaSide, because rootSide is empty for these DNs.
8. groom pools the whole DNg12_* family as putative head-grooming DNs. Targets are restricted to VNC MNs; their strongest outputs are neck (MNnm*) and front-leg MNs. Allowing brain cb_motor targets instead yields mostly direct DNg12 -> neck-MN (CvN4/GNG314) contacts.
9. feed uses Fdg = MaleCNS type GNG588, found through its 'Shiu 2022: Fdg' synonym. It is an SEZ interneuron, not a DN; paths target brain (cb_motor) proboscis/pharyngeal MNs (e.g. MN9, rostrum protractor).
10. Looming uses all LPLC2 and LC4 (both hemispheres). Paths into DNp01: per type, the 2 strongest direct VPN->GF synapses (one per GF side) plus the 2 strongest 2-3 hop routes (distinct intermediate types), ranked by bottleneck. Single-VPN direct contacts are modest (about 60-90 synapses), but the populations converge heavily on the GF (`direct_w_to_DNp01` sums over all bodies).
11. Bodies without a soma position were dropped from named groups (positioned/total): DNg12_d 1/2.
12. The KC pool is the 4,054 positioned Kenyon cells (of 4,064 traced); KCs without a soma position are excluded from the k-WTA.
13. Orientation: +x fly's right, +y anterior (brain up, VNC hangs down), +z dorsal. "Anterior facing +z" cannot coexist with "VNC below" because the VNC lies posterior to the brain in this volume; a camera on +z sees the CNS from above (dorsal view).

## `build_vision.py` outputs (`data/`)

| File | Size | Content |
|---|---|---|
| `vision.json` | 413,052 B | eye columns, LC/LPLC targets, visuomotor drive, notes (schema below) |
| `vision_w.bin` | 3,581,856 B | sparse column -> target weights, little-endian: `int32 col[n]`, then `int32 tgt[n]`, then `float32 w[n]`, with `n = vision.json.n_weights` (298,488), so 12 x n bytes |

Build steps: take every Traced/Anchor neuron with an eye-column hex assignment (`assignedOlHex1/2`) and a left/right soma side; one column per (side, hex1, hex2). Fit viewing directions from lamina (L1/L2/L3/L5) soma positions, then stretch them to a typical fly eye field (azimuth -15..165 deg per eye, + = the eye's own side; elevation -60..70 deg). Targets are all positioned `LC*` / `LPLC*` neurons. Column -> target weights are real synapses, direct plus one optic-lobe intermediate, each signed by the presynaptic neuron's consensus neurotransmitter (ACh +1; GABA, glutamate, histamine -1; others 0) and divided by the postsynaptic neuron's total input. Only the 64 strongest inputs per target are kept, L1-normalised per target.

### `vision.json` schema

- `columns`: parallel arrays, one entry per eye column (1,771: 879 left, 892 right).
  - `side` (`"L"`/`"R"`), `hex` (`[h1, h2]`), `az`, `el` (degrees, robot frame).
  - `light`: up to 8 positioned column-neuron indices (into `brain_positions.f32`) to light in the brain view.
- `targets`: parallel arrays, one entry per LC/LPLC neuron (4,664 neurons, 51 types; all receive column input).
  - `idx` (brain index), `type` (index into `typeNames`), `typeNames`, `side`, `rf_az`, `rf_el` (receptive-field centre, weighted by positive inputs), `has_input`.
- `n_weights`: number of entries in `vision_w.bin` (298,488).
- `drive`: innate visuomotor wiring from each target to five motor programs.
  - `actions`: `walk_forward`, `veer_left`, `veer_right`, `walk_backward`, `takeoff` (command neurons from `circuits.json` actions `walk_forward`, `turn_left`, `turn_right`, `walk_backward`, `takeoff`: DNp09, DNa02 L/R, MDN, DNp01).
  - `w`: targets x 5, signed synapse-derived weights (direct target->DN plus target->any neuron->DN), each column scaled so that all targets active at once gives 1.
- `notes`: 6 strings describing the approximations (fitted eye angles; the plastic runtime gains are a modelling choice; static linear read-through, no dynamics or direction selectivity).
