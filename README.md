# Fly x Jev

A browser demo of a small robot walking down a road, driven by wiring from the real MaleCNS v1.0 fruit fly connectome. The robot's compound eye is laid out on the fly's 1,771 real eye columns, whose signal passes through real synapses to 4,664 LC/LPLC visual projection neurons. Smells go through real projection-neuron to Kenyon-cell to MBON wiring in the mushroom body, where dopamine (PPL1 for pain, PAM for sugar) weakens KC->MBON synapses and so forms a memory. Motor output is a choice between descending neurons (DNp09 walk, DNa02 left/right veer, MDN back away, DNp01 giant-fiber takeoff) and the Fdg feeding neuron. TypeSafe's Jev model (System One API) reads the robot's senses, the mushroom-body output and the visual pathway drive, and picks which of those neurons fires next, several times a second. Next to the scene, a three.js point cloud of 141,208 neurons shows which real cells and paths are involved.

## Quick start

1. Get a TypeSafe API key (the demo calls `POST https://api.typesafe.ai/v1/systemone`).
2. Configure and run (Python 3.9+, standard library only):

   ```
   cp .env.example .env        # then set TYPESAFE_API_KEY=...
   python3 arena/server.py
   ```

3. Open http://127.0.0.1:8777 and press Play, or open http://127.0.0.1:8777/?demo to run the scripted demo straight away.

`arena/data/` is committed prebuilt. The MaleCNS download (about 560 MB) is only needed to rebuild it; see [Rebuilding the data](#rebuilding-the-data).

Configuration is read from the environment first, then from `.env` at the repo root (`.env` is outside `arena/`, the directory the server serves, and is gitignored):

| Variable | Default | |
|---|---|---|
| `TYPESAFE_API_KEY` | none | required for decisions |
| `JEV_MODEL` | `jev-latest` | model name sent to the API |
| `PORT` | `8777` | the server binds to 127.0.0.1 only |

The key never reaches the browser: `server.py` serves the static files and proxies `POST /api/decide` to TypeSafe with a small keep-alive connection pool. `GET /api/health` returns `{"jev_key": true|false, "model": ...}`. Without a key the page still loads, and `/api/decide` returns 503 with an explanation; the robot then keeps its last action and the HUD shows `jev offline`.

The page loads three.js r170 from jsDelivr and fonts from Google Fonts, so it needs internet access.

## Controls

Dock buttons:

- **Play / Pause**: start or stop the simulation (the page starts paused). Jev is only called while playing.
- **Demo**: resets and runs the scripted sequence: toxic waste on the road (first contact, PPL1 dopamine), honey (feeding, PAM dopamine), toxic waste again (now avoided), honey off to the side (now sought), then a full barrier across the road, twice (the second time the takeoff pathway has been strengthened).
- **Reset**: clears objects and the trail, and resets the mushroom-body synapses, the visuomotor gains and the operant motor memory to naive.
- **Honey**, **Toxic waste**, **Barrier**: arm a tool, then click the road to drop it there (dropping also unpauses). A barrier is a row of five toxic lumps across the whole road.
- **Brain**: open or close the connectome popup. It also opens by itself for a few seconds on dopamine events and takeoffs.

Keyboard:

| Key | Action |
|---|---|
| `1` / `2` / `3` | arm Honey / Toxic waste / Barrier (press again to disarm) |
| `Esc` | disarm the current tool |
| `Space` | run the demo |
| `P` | play / pause |
| `B` | brain popup |
| `H` | hide the dock (for screen recording) |

URL parameter: `?demo` starts the demo as soon as the data has loaded.

## How it works

Each frame (`arena/js/main.js`) the world is stepped, the robot's senses are read, and the three model pieces below are updated. In parallel, `arena/js/jev.js` sends the current state to Jev and applies the returned action.

### Eye -> LC/LPLC (`js/eye.js`, `build_vision.py`)

Every one of the 1,771 real eye columns (879 left, 892 right) casts one ray from the robot's head at 20 Hz. Dark things (toxic lumps, curbs) and luminance change drive the column. Column activity is pushed through precomputed sparse weights (`data/vision_w.bin`: real column-neuron synapses to the 4,664 LC/LPLC neurons, direct or via one optic-lobe intermediate, signed by predicted neurotransmitter) and low-pass filtered into a firing rate per LC/LPLC neuron. Jev receives this as nine numbers (0-9), LC/LPLC activity binned by receptive-field direction from far left to far right. LPLC2/LC4 activity is also averaged per eye as a looming signal, which flashes the LPLC2/LC4 -> DNp01 paths in the brain view.

### Mushroom body (`js/mb.js`, `build_circuits.py`)

Two odors exist: A (toxic waste) and B (honey). Each odor activates a fixed set of 203 Kenyon cells chosen from real PN->KC synapse counts. Four MBON types are read out: MBON11 and MBON12 (approach), MBON01 and MBON04 (avoid). Each active KC's synapse onto an MBON has a gain that starts at 1.

- Punishment dopamine (PPL1 types PPL101, PPL103) targets the approach MBONs; reward dopamine (PAM types PAM01, PAM02, PAM05, PAM06) targets the avoid MBONs. The DAN -> MBON pairing comes from `circuits.json`.
- While dopamine is present, the gain of every KC active for the odor currently smelled is multiplied by `exp(-0.38 * dopamine * concentration * dt)`. Gains relax back toward 1 with a 420 s time constant.
- An odor's valence is the mean approach-MBON strength minus the mean avoid-MBON strength (relative to naive), clipped to -1..1. Jev only sees the label: `aversive` (<= -0.08), `attractive` (>= 0.08) or `unknown`.

Punishment dopamine comes only from toxic contact; reward dopamine comes from feeding on honey.

### Pain

`js/world.js` has several pain sources, but only one of them is paired with smell:

| Source | Level | Goes to |
|---|---|---|
| Touching a toxic lump (including barrier lumps) | 1.0 | mushroom body (PPL1 punishment), visuomotor learning, Jev `pain` |
| Hitting the road border (curb) | 0.8 | visuomotor learning, Jev `pain` and `curb` |
| Drifting off the centre line | 0 on the line, up to 0.4 at the border | Jev `center_line` text and HUD only |

Border and off-centre pain never reach the mushroom body, so they do not change any smell's valence.

### Visuomotor pathway learning (`js/visuomotor.js`)

`vision.json` also holds, for each LC/LPLC neuron, its real signed wiring to the command neurons of five motor programs (walk forward, veer left, veer right, back away, takeoff), direct or through one intermediate neuron. At runtime a plastic gain sits on each (LC/LPLC type, motor program) pathway; the takeoff pathway starts at 0.5, the rest at 1. Pathways driving the current action are tagged with an eligibility trace (1.5 s). Pain above 0.3, or being stalled for more than 1.5 s, weakens tagged pathways; moving on after being stuck strengthens the pathway of the action that got through. Gains are clamped to 0.12..4 and relax toward baseline over 300 s. Jev receives the resulting drive per command neuron as `descending_drive` (0-9). This is how the barrier is learned: walking into it hurts, veering and backing get nowhere, and once takeoff breaks through, the LC4/LPLC2 -> DNp01 pathway is stronger the next time.

### Jev as action selection (`js/jev.js`)

Jev is asked one `choice` question per call with six options:

| Action | Neuron | Circuit in `circuits.json` |
|---|---|---|
| `walk_forward` | DNp09 | walk_forward |
| `veer_left` | left DNa02 | turn_left |
| `veer_right` | right DNa02 | turn_right |
| `walk_backward` | MDN | walk_backward |
| `takeoff` | DNp01 (giant fiber) | takeoff |
| `feed` | Fdg (GNG588) | feed |

The state it gets:

- `currently_doing`: current program and how long it has run.
- `hungry`: fed level below 0.7.
- `sugar_under_feet`.
- `pain`: `strong`, `mild` or `none`.
- `curb`: on your LEFT / on your RIGHT / not touching.
- `smell_A`, `smell_B`: `none`, or `{stronger_on, means}` where `means` is the MBON-derived valence label.
- `stuck`: no forward progress for more than 1.5 s.
- `vision`: the nine LC/LPLC direction levels.
- `descending_drive`: learned visuomotor drive per command neuron.
- `center_line`: on it, or to your LEFT/RIGHT with the off-centre pain percentage.

It never receives object labels, pain history or the training protocol; what an object means reaches Jev only through the mushroom-body valence. The instructions are eight hand-written, first-match rules in `js/jev.js` (feed on sugar when hungry, back away from strong pain, veer off the curb, veer away from aversive smells, toward attractive ones when hungry, follow `descending_drive` when stuck, walk forward to investigate unknown things, otherwise follow the centre line). When the robot has been stalled for more than 1.5 s, the action is sampled from Jev's returned probabilities instead of taking its top choice.

Calls are made only while playing and while the tab is rendering: at most two requests in flight and at least 160 ms between sends, so the decision rate depends on API latency. Decisions that arrive out of order are dropped. The HUD shows latency, decision count and an estimated spend.

## Real vs modelled

Real, from the MaleCNS v1.0 connectome:

- The 141,208 neurons shown, their soma positions and superclasses (Traced/Anchor neurons with a soma location).
- Synapse counts: PN->KC (which KCs each odor drives), KC->MBON (the plastic weights), DAN->MBON (which dopamine reaches which MBON).
- Cell types and identities: MBON, PPL1/PAM, DNp09, DNa02, MDN, DNp01, Fdg (GNG588), LC/LPLC types, eye-column hex assignments.
- Anatomical paths from each descending neuron to motor neurons, and LPLC2/LC4 -> DNp01.
- Column -> LC/LPLC and LC/LPLC -> descending-neuron synapse weights, signed by the dataset's neurotransmitter predictions.

Modelled:

- Odors are synthetic: seed-7 random draws of 6 glomeruli each. The KC code is a static top 5% (k = 203) of summed PN->KC drive, not a simulation of APL inhibition.
- The plasticity rules in `js/mb.js` and `js/visuomotor.js`, and all their constants.
- MBON approach/avoid roles come from the literature (Aso et al. 2014), not from the connectome.
- Eye-column viewing angles are fitted from lamina geometry and then stretched to a typical fly eye field; absolute angles are approximate.
- The robot body, physics, road, objects and motor programs (`js/world.js`, `js/robots.js`, `js/track.js`).
- Jev's rules are hand-written in `js/jev.js`: the innate behaviour (turn from aversive smells, back away from pain, and so on) is prompt text, not wiring.
- There is no spiking or dynamical simulation. The pipeline is a linear read-through of the wiring: signal is multiplied through static weights, with simple rate filtering. It cannot reproduce motion-direction selectivity; luminance change is used as an input feature instead. The paths flashed in the brain view are anatomical routes, not simulated activity.

`arena/DATA.md` lists every approximation made by the build scripts.

## Rebuilding the data

Only needed if you change the build scripts. Requires `pyarrow`, `pandas`, `numpy` and `scipy`, plus three files (about 560 MB) from the MaleCNS v1.0 release:

```
mkdir -p male-cns-data && cd male-cns-data
BASE=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
curl -LO $BASE/body-annotations-male-cns-v1.0-minconf-0.5.feather
curl -LO $BASE/connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather
curl -LO $BASE/body-neurotransmitters-male-cns-v1.0.feather
cd ..
python3 arena/build_circuits.py
python3 arena/build_vision.py
```

The input directory defaults to `male-cns-data/` at the repo root. Override it with `--malecns-dir PATH` or `MALECNS_DIR=PATH`. See `arena/DATA.md` for the files, output schemas and build choices.

## File map

```
.env.example            configuration template (copy to .env)
arena/
  server.py             static server + Jev proxy (stdlib only)
  index.html            page layout: scene, readout card, compound-eye card, dock, brain popup
  style.css
  build_circuits.py     MaleCNS -> brain_*.{f32,u8,json}, circuits.json
  build_vision.py       MaleCNS -> vision.json, vision_w.bin
  DATA.md               data inputs, output schemas, approximations
  data/                 prebuilt outputs of the two build scripts
  js/main.js            boot, per-frame loop, HUD, demo script, controls
  js/world.js           road, objects, smell fields, pain, body physics, motor programs
  js/track.js           road / ground rendering (used by world.js)
  js/robots.js          robot bodies; world.js uses Strider
  js/eye.js             compound eye: ray casting, column -> LC/LPLC propagation, readout
  js/mb.js              mushroom body: KC->MBON gains, dopamine plasticity, valence
  js/visuomotor.js      plastic LC/LPLC -> descending-neuron pathways
  js/jev.js             Jev state, question, action list, request loop
  js/brain.js           connectome point cloud, flashes, path animation
  robots.html           standalone viewer for the four robot bodies (not linked from index.html)
```

## Credits and citation

Connectome data: MaleCNS v1.0, a collaboration of FlyEM (HHMI Janelia Research Campus), the University of Cambridge (Department of Zoology), the MRC Laboratory of Molecular Biology and Google Research. The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); files in `arena/data/` are derived from it (selected neurons, positions, synapse counts and derived weights). Project page: https://male-cns.janelia.org/.

If you use this work, cite the dataset paper:

> Berg S, Beckett IR, Costa M, et al. Sexual dimorphism in the complete Drosophila male central nervous system connectome. *Cell* 189(18):5504-5526.e15 (2026). https://doi.org/10.1016/j.cell.2026.08.015

Neuron roles used by the model come from: Aso et al. 2014 (MBON valence), Bidaye et al. 2014 (MDN), Bidaye et al. 2020 (DNp09), Rayshubskiy et al. 2020 (DNa02), Flood et al. 2013 (Fdg).

Jev is a model by TypeSafe, used through its API. This repository contains no TypeSafe code.

## License

Code: MIT, see [LICENSE](LICENSE). The derived connectome data in `arena/data/` remains under the MaleCNS CC BY 4.0 license and requires the attribution above.
