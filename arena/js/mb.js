// Mushroom body associative memory on real MaleCNS KC→MBON weights.
// Dopamine (PPL1 = punishment, PAM = reward) depresses the synapses from the
// Kenyon cells that are active *right now* onto the MBONs in its compartment.
// Jev never sees "odor A was shocked" — only the MBON output these weights produce.

const ETA = 0.38;        // depression rate per second of full DA + full odor
const TAU = 420;         // seconds for synapses to relax back (forgetting)

export class MushroomBody {
  constructor(circuits) {
    this.c = circuits;
    this.mbons = circuits.mbons;
    this.approach = this.mbons.filter((m) => m.role === 'approach').map((m) => m.type);
    this.avoid = this.mbons.filter((m) => m.role === 'avoid').map((m) => m.type);
    // Plastic gain per (MBON type, KC index): starts at 1 (naive).
    this.gain = {};
    this.naive = {};                                   // naive summed weight per MBON per odor
    for (const m of this.mbons) {
      this.gain[m.type] = new Map();
      this.naive[m.type] = {};
      const byOdor = circuits.kc_mbon[m.type] || {};
      for (const o of Object.keys(circuits.odors)) {
        const e = byOdor[o] || { kc: [], w: [] };
        this.naive[m.type][o] = e.w.reduce((a, b) => a + b, 0) || 1;
        for (const k of e.kc) this.gain[m.type].set(k, 1);
      }
    }
    // Which MBONs each dopamine signal reaches (from the DAN→MBON connectivity).
    this.targets = { punishment: new Set(), reward: new Set() };
    for (const d of circuits.dans) for (const t of d.targets || []) this.targets[d.signal]?.add(t);
    if (!this.targets.punishment.size) this.approach.forEach((t) => this.targets.punishment.add(t));
    if (!this.targets.reward.size) this.avoid.forEach((t) => this.targets.reward.add(t));
    this.da = { punishment: 0, reward: 0 };
  }

  // Current weight of an odor's KC ensemble onto one MBON type, relative to naive.
  strength(mbon, odor) {
    const e = this.c.kc_mbon[mbon]?.[odor];
    if (!e) return 1;
    const g = this.gain[mbon];
    let s = 0;
    for (let i = 0; i < e.kc.length; i++) s += e.w[i] * g.get(e.kc[i]);
    return s / this.naive[mbon][odor];
  }

  valence(odor) {
    const mean = (list) => list.length ? list.reduce((a, t) => a + this.strength(t, odor), 0) / list.length : 1;
    const app = mean(this.approach), av = mean(this.avoid);
    return { approach: app, avoid: av, value: Math.max(-1, Math.min(1, app - av)) };
  }

  // MBON output for what the fly smells now (conc: {A, B} in 0..1).
  output(conc) {
    const out = {};
    for (const m of this.mbons) {
      let s = 0, n = 0;
      for (const o of Object.keys(conc)) { s += conc[o] * this.strength(m.type, o); n += conc[o]; }
      out[m.type] = n > 0.02 ? s / Math.max(1, n) : 0;
    }
    return out;
  }

  // da: {punishment, reward} in 0..1; conc: odors at the fly.
  step(dt, conc, da) {
    this.da = da;
    for (const signal of ['punishment', 'reward']) {
      const d = da[signal];
      if (d <= 0.01) continue;
      for (const mbon of this.targets[signal]) {
        const g = this.gain[mbon];
        if (!g) continue;
        for (const o of Object.keys(conc)) {
          if (conc[o] < 0.05) continue;
          const e = this.c.kc_mbon[mbon]?.[o];
          if (!e) continue;
          const f = Math.exp(-ETA * d * conc[o] * dt);
          for (const k of e.kc) g.set(k, g.get(k) * f);
        }
      }
    }
    const relax = dt / TAU;
    for (const g of Object.values(this.gain)) for (const [k, v] of g) if (v < 1) g.set(k, v + (1 - v) * relax);
  }

  reset() { for (const g of Object.values(this.gain)) for (const k of g.keys()) g.set(k, 1); }
}

export function valenceLabel(v) {
  if (v <= -0.6) return 'strongly aversive';
  if (v <= -0.25) return 'aversive';
  if (v <= -0.08) return 'slightly aversive';
  if (v < 0.08) return 'neutral';
  if (v < 0.25) return 'slightly attractive';
  if (v < 0.6) return 'attractive';
  return 'strongly attractive';
}
