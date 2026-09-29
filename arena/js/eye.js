// The robot's compound eye, laid out on the fly's real eye columns (MaleCNS hex
// assignments). Each column casts one ray from the head; what it sees drives
// that column's neurons, and the signal flows through real, signed synapses to
// the LC/LPLC visual projection neurons. Jev reads those neurons, not the world.

const DEG = Math.PI / 180;
const HAZE_NEAR = 1.2, HAZE_FAR = 7;        // contrast falls off with distance
const LUM = { sky: 0.96, runway: 0.97, ground: 0.86, curb: 0.8, honey: 0.62, toxic: 0.06, horizon: 0.93 };
export const DIR_BINS = 9;                  // readout: far left (+80°) … far right (−80°)
const BIN_EDGE = 80;

export class Eye {
  constructor(vision, wbuf) {
    const c = vision.columns, t = vision.targets;
    this.N = c.side.length;
    this.T = t.idx.length;
    this.side = c.side; this.hex = c.hex; this.az = c.az; this.el = c.el; this.light = c.light;
    this.targets = t;
    this.notes = vision.notes;
    // robot-local ray directions (x = left, y = up, z = forward)
    this.dir = new Float32Array(this.N * 3);
    for (let i = 0; i < this.N; i++) {
      const a = c.az[i] * DEG, e = c.el[i] * DEG;
      this.dir.set([Math.sin(a) * Math.cos(e), Math.sin(e), Math.cos(a) * Math.cos(e)], i * 3);
    }
    // sparse weights, grouped by column for a fast push
    const n = vision.n_weights;
    const col = new Int32Array(wbuf, 0, n), tgt = new Int32Array(wbuf, n * 4, n), w = new Float32Array(wbuf, n * 8, n);
    const counts = new Int32Array(this.N + 1);
    for (let k = 0; k < n; k++) counts[col[k] + 1]++;
    for (let i = 0; i < this.N; i++) counts[i + 1] += counts[i];
    this.ptr = counts;
    this.wt = new Int32Array(n); this.ww = new Float32Array(n);
    const fill = counts.slice(0, this.N);
    for (let k = 0; k < n; k++) { const p = fill[col[k]]++; this.wt[p] = tgt[k]; this.ww[p] = w[k]; }

    this.lum = new Float32Array(this.N).fill(LUM.runway);
    this.prev = new Float32Array(this.N).fill(LUM.runway);
    this.act = new Float32Array(this.N);        // column drive
    this.input = new Float32Array(this.T);
    this.rate = new Float32Array(this.T);       // LC/LPLC activity 0..~1
    this.clock = 0;
    this.loomHist = [];

    // groups for readout
    const names = t.typeNames;
    const isType = (re) => t.type.map((k) => re.test(names[k]));
    this.isLoom = isType(/^(LPLC2|LC4)$/);
    this.bin = t.rf_az.map((a) => Math.max(0, Math.min(DIR_BINS - 1, Math.floor(((BIN_EDGE - a) / (2 * BIN_EDGE)) * DIR_BINS))));
  }

  // ------------------------------------------------------------ rays
  cast(world) {
    const b = world.body, h = b.heading, ch = Math.cos(h), sh = Math.sin(h);
    const hx = b.x + sh * 0.1, hz = b.z + ch * 0.1, hy = 0.065 + (b.alt || 0);
    const road = world.road, curbH = 0.035;
    const lumps = [], puddles = [];
    for (const o of world.objects) {
      if (!o.landed) continue;
      if (o.solid) lumps.push([o.x, o.r * 0.32, o.z, o.r * 0.82]);
      else if (o.amount > 0.05) puddles.push([o.x, o.z, o.r * Math.sqrt(o.amount)]);
    }
    for (let i = 0; i < this.N; i++) {
      const lx = this.dir[i * 3], ly = this.dir[i * 3 + 1], lz = this.dir[i * 3 + 2];
      // robot frame → world: forward (sin h, cos h), left (cos h, −sin h)
      const dx = lx * ch + lz * sh, dy = ly, dz = -lx * sh + lz * ch;
      let t = Infinity, lum = LUM.sky + 0.02 * Math.min(1, dy * 2);
      if (dy < -1e-4) {                                     // floor
        const tg = -hy / dy, gx = hx + dx * tg, gz = hz + dz * tg;
        t = tg; lum = Math.abs(gx) < road ? LUM.runway : LUM.ground;
        for (const [px, pz, pr] of puddles) if ((gx - px) ** 2 + (gz - pz) ** 2 < pr * pr) { lum = LUM.honey; break; }
      }
      if (Math.abs(dx) > 1e-4) {                            // inner curb faces
        const tc = ((dx > 0 ? road : -road) - hx) / dx;
        if (tc > 0 && tc < t && hy + dy * tc < curbH) { t = tc; lum = LUM.curb; }
      }
      for (const [cx, cy, cz, cr] of lumps) {               // toxic lumps as spheres
        const ox = hx - cx, oy = hy - cy, oz = hz - cz;
        const bq = ox * dx + oy * dy + oz * dz, cq = ox * ox + oy * oy + oz * oz - cr * cr;
        const disc = bq * bq - cq;
        if (disc > 0) { const tl = -bq - Math.sqrt(disc); if (tl > 0 && tl < t) { t = tl; lum = LUM.toxic; } }
      }
      if (t < Infinity) {
        const haze = Math.min(1, Math.max(0, (t - HAZE_NEAR) / (HAZE_FAR - HAZE_NEAR)));
        lum += (LUM.horizon - lum) * haze;
      }
      this.lum[i] = lum;
    }
  }

  // ------------------------------------------------------------ neurons
  update(dt, world) {
    this.clock += dt;
    this.prev.set(this.lum);
    this.cast(world);
    const input = this.input;
    input.fill(0);
    for (let i = 0; i < this.N; i++) {
      const off = Math.max(0, 0.88 - this.lum[i]);                       // dark things
      const change = Math.abs(this.lum[i] - this.prev[i]) / Math.max(dt, 1e-3) * 0.08;
      const a = Math.min(1.5, off + change);
      this.act[i] += (a - this.act[i]) * Math.min(1, dt * 12);
      const ai = this.act[i];
      if (ai < 0.01) continue;
      for (let p = this.ptr[i]; p < this.ptr[i + 1]; p++) input[this.wt[p]] += ai * this.ww[p];
    }
    const k = Math.min(1, dt * 8);                                       // ~125 ms integration
    for (let j = 0; j < this.T; j++) {
      const target = Math.max(0, input[j] - 0.02) * 2.2;
      this.rate[j] += (Math.min(1.2, target) - this.rate[j]) * k;
    }
    const loom = this.loom();
    this.loomHist.push({ t: this.clock, L: loom.L, R: loom.R });
    while (this.loomHist.length && this.loomHist[0].t < this.clock - 0.6) this.loomHist.shift();
  }

  loom() {
    let L = 0, R = 0, nl = 0, nr = 0;
    const side = this.targets.side;
    for (let j = 0; j < this.T; j++) {
      if (!this.isLoom[j]) continue;
      if (side[j] === 'L') { L += this.rate[j]; nl++; } else { R += this.rate[j]; nr++; }
    }
    return { L: nl ? L / nl : 0, R: nr ? R / nr : 0 };
  }

  // What Jev gets: LC/LPLC activity by receptive-field direction, plus looming per eye.
  readout() {
    // each direction is read from its most active neurons (top 12%), like a downstream cell pooling its strongest inputs
    const byBin = Array.from({ length: DIR_BINS }, () => []);
    for (let j = 0; j < this.T; j++) byBin[this.bin[j]].push(this.rate[j]);
    const levels = byBin.map((r) => {
      if (!r.length) return 0;
      r.sort((a, b) => b - a);
      const k = Math.max(3, Math.round(r.length * 0.12));
      let s = 0; for (let i = 0; i < k; i++) s += r[i];
      return Math.min(9, Math.round((s / k) * 11));
    });
    const now = this.loom(), past = this.loomHist[0] || { L: now.L, R: now.R };
    const word = (v) => (v > 0.35 ? 'high' : v > 0.15 ? 'moderate' : v > 0.05 ? 'low' : 'silent');
    const trend = (a, b) => (a - b > 0.08 ? 'rising fast' : a - b > 0.02 ? 'rising' : b - a > 0.02 ? 'falling' : 'steady');
    return {
      levels,
      loom: { left_eye: word(now.L), right_eye: word(now.R), left_trend: trend(now.L, past.L), right_trend: trend(now.R, past.R) },
      raw: now,
    };
  }
}
