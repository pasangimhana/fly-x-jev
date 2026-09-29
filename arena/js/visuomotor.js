// Plastic visuomotor pathways on real wiring. Each LC/LPLC neuron reaches the
// command neurons of five motor programs through real synapses (build_vision.py);
// a learnable gain sits on every (visual type -> command) pathway. Pathways that
// drove the current action are tagged for a moment (eligibility); pain and being
// stuck weaken tagged pathways, breaking through strengthens them. The wiring is
// real, the learning rule is a modelling choice.

const TAU_TRACE = 1.5;        // s, how long a pathway stays tagged after driving an action
const TAU_RELAX = 300;        // s, gains drift back toward baseline
const GAIN_MIN = 0.12, GAIN_MAX = 4;
const PAIN_RATE = 2.0;        // per second of pain, times the tag
const STALL_RATE = 1.2;      // per second stuck (no forward progress), times the tag
const SUCCESS = 3.2;          // impulse when progress resumes after being stuck

export const DRIVE_LABEL = { walk_forward: 'DNp09 walk', veer_left: 'DNa02-L veer left', veer_right: 'DNa02-R veer right', walk_backward: 'MDN back away', takeoff: 'DNp01 boosters' };

export class Visuomotor {
  constructor(vision) {
    const d = vision.drive, t = vision.targets;
    this.actions = d.actions;
    this.A = d.actions.length;
    this.T = t.idx.length;
    this.typeNames = t.typeNames;
    this.K = t.typeNames.length;
    this.type = Int32Array.from(t.type);
    this.W = new Float32Array(this.T * this.A);
    d.w.forEach((row, i) => row.forEach((v, a) => { this.W[i * this.A + a] = v; }));
    // baseline: the robot's boosters are new hardware, so the giant-fiber pathway starts weakly coupled
    this.base = new Float32Array(this.K * this.A).fill(1);
    const ti = this.actions.indexOf('takeoff');
    for (let k = 0; k < this.K; k++) this.base[k * this.A + ti] = 0.5;
    this.gain = Float32Array.from(this.base);
    this.trace = new Float32Array(this.K * this.A);
    this.contrib = new Float32Array(this.K * this.A);
    this.drive = new Float32Array(this.A);
    this.stuck = { on: false, z: 0 };
    this.onEvent = () => {};
  }

  // rate: LC/LPLC activity (Eye.rate); action: motor program running now; body: world.body; stalled: s without progress
  update(dt, rate, action, { pain, stalled, z }) {
    const A = this.A, c = this.contrib;
    c.fill(0);
    for (let i = 0; i < this.T; i++) {
      const r = rate[i];
      if (r < 0.01) continue;
      const k = this.type[i] * A;
      for (let a = 0; a < A; a++) { const w = this.W[i * A + a]; if (w > 0) c[k + a] += r * w; }
    }
    this.drive.fill(0);
    for (let k = 0; k < this.K; k++) for (let a = 0; a < A; a++) this.drive[a] += c[k * A + a] * this.gain[k * A + a];

    // eligibility: tag the pathways that are driving the action being executed
    const decay = Math.exp(-dt / TAU_TRACE), ai = this.actions.indexOf(action);
    for (let j = 0; j < this.trace.length; j++) this.trace[j] *= decay;
    if (ai >= 0) for (let k = 0; k < this.K; k++) this.trace[k * A + ai] += c[k * A + ai] * dt * 3;

    // three-factor learning: (pathway tag) x (outcome signal)
    let signal = 0;
    if (pain > 0.3) signal -= PAIN_RATE * pain;
    if (stalled > 1.5) signal -= STALL_RATE;
    if (signal) this.apply(signal * dt);
    if (stalled > 1.5 && !this.stuck.on) this.stuck = { on: true, z, action };
    if (this.stuck.on && z > this.stuck.z + 0.6) {             // broke through: reward what just happened
      this.stuck.on = false;
      const before = this.pathGain(action);
      this.apply(SUCCESS);
      this.onEvent({ type: 'strengthened', action, before, after: this.pathGain(action) });
    }
    // slow relaxation toward baseline
    const rel = dt / TAU_RELAX;
    for (let j = 0; j < this.gain.length; j++) this.gain[j] += (this.base[j] - this.gain[j]) * rel;
  }

  apply(amount) {
    for (let j = 0; j < this.gain.length; j++) {
      if (this.trace[j] < 1e-4) continue;
      this.gain[j] = Math.min(GAIN_MAX, Math.max(GAIN_MIN, this.gain[j] * Math.exp(amount * this.trace[j])));
    }
  }

  // effective gain of one motor program's visual pathway, weighted by how much real wiring each type contributes
  pathGain(action) {
    const a = this.actions.indexOf(action);
    if (a < 0) return 1;
    let num = 0, den = 0;
    const wt = new Float32Array(this.K);
    for (let i = 0; i < this.T; i++) { const w = this.W[i * this.A + a]; if (w > 0) wt[this.type[i]] += w; }
    for (let k = 0; k < this.K; k++) { num += wt[k] * this.gain[k * this.A + a]; den += wt[k]; }
    return den ? num / den : 1;
  }

  // strongest real visual types feeding an action (for labels)
  topTypes(action, n = 2) {
    const a = this.actions.indexOf(action), wt = new Map();
    for (let i = 0; i < this.T; i++) { const w = this.W[i * this.A + a]; if (w > 0) wt.set(this.type[i], (wt.get(this.type[i]) || 0) + w); }
    return [...wt.entries()].sort((x, y) => y[1] - x[1]).slice(0, n).map(([k]) => this.typeNames[k]);
  }

  // what Jev gets: learned drive per command neuron, 0-9
  readout() {
    const out = {};
    this.actions.forEach((a, i) => { out[DRIVE_LABEL[a]] = Math.min(9, Math.round(this.drive[i] * 14)); });
    return out;
  }

  reset() { this.gain.set(this.base); this.trace.fill(0); this.stuck.on = false; }
}
