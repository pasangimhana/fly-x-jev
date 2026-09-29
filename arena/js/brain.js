// The MaleCNS connectome as graphite stipple on white: ~141k somata as faint
// ink dots; the circuits the fly is using light up in ink, honey or accent.
import * as THREE from 'three';

export const BCOL = {
  vision: new THREE.Color('#4a6fa5'),    // eye columns and visual projection neurons
  odorA: new THREE.Color('#1c1c1c'),     // toxic smell
  odorB: new THREE.Color('#c98a1c'),     // honey smell
  approach: new THREE.Color('#1c1c1c'),
  avoid: new THREE.Color('#1c1c1c'),
  ppl1: new THREE.Color('#E55B2B'),      // punishment dopamine
  pam: new THREE.Color('#c98a1c'),       // reward dopamine
  dn: new THREE.Color('#E55B2B'),
  loom: new THREE.Color('#E55B2B'),
  motor: new THREE.Color('#E55B2B'),
};

const MAX_SPARKS = 80, SPARK_SEGS = 10;

export class Brain {
  constructor(canvas, labelsEl) {
    this.canvas = canvas;
    this.labelsEl = labelsEl;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setClearColor('#ffffff');
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#ffffff');
    this.camera = new THREE.PerspectiveCamera(30, 1, 0.01, 50);
    this.time = 0;
    this.hot = new Map();
    this.sustained = [];
    this.touched = [];
    this.sparks = [];
    this.labels = [];
    this.visible = false;
    this.view = { center: new THREE.Vector3(), dist: 3, elev: 0.1 };
    this.viewTarget = { center: new THREE.Vector3(), dist: 3, elev: 0.1 };
  }

  async load(base = 'data/') {
    const [pos, cls, meta, circuits] = await Promise.all([
      fetch(base + 'brain_positions.f32').then((r) => r.arrayBuffer()),
      fetch(base + 'brain_superclass.u8').then((r) => r.arrayBuffer()),
      fetch(base + 'brain_meta.json').then((r) => r.json()),
      fetch(base + 'circuits.json').then((r) => r.json()),
    ]);
    this.pos = new Float32Array(pos);
    this.cls = new Uint8Array(cls);
    this.meta = meta;
    this.circuits = circuits;
    this.N = this.pos.length / 3;
    this.buildPoints();
    this.buildSparks();
    this.buildMemoryFan();
    this.computeViews();
    this.buildLabels();
    return circuits;
  }

  P(i) { return new THREE.Vector3(this.pos[i * 3], this.pos[i * 3 + 1], this.pos[i * 3 + 2]); }
  centroid(idx) {
    const c = new THREE.Vector3();
    if (!idx?.length) return c;
    for (const i of idx) c.add(this.P(i));
    return c.divideScalar(idx.length);
  }

  buildPoints() {
    const N = this.N, seed = new Float32Array(N);
    for (let i = 0; i < N; i++) seed[i] = Math.random();
    this.act = new Float32Array(N);
    this.tint = new Float32Array(N * 3);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    geo.setAttribute('seed', new THREE.BufferAttribute(seed, 1));
    geo.setAttribute('act', new THREE.BufferAttribute(this.act, 1).setUsage(THREE.DynamicDrawUsage));
    geo.setAttribute('tint', new THREE.BufferAttribute(this.tint, 3).setUsage(THREE.DynamicDrawUsage));
    this.pointU = { uPx: { value: 1 } };
    this.points = new THREE.Points(geo, new THREE.ShaderMaterial({
      uniforms: this.pointU, transparent: true, depthWrite: false,
      vertexShader: /* glsl */`
        attribute vec3 tint; attribute float act; attribute float seed;
        uniform float uPx;
        varying vec3 vC; varying float vA;
        void main(){
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          float a = clamp(act, 0.0, 1.5);
          float on = smoothstep(0.0, 0.3, a);
          vC = mix(vec3(0.11), tint, on);
          vA = mix(0.12 + 0.06 * seed, 0.95, on);
          gl_PointSize = max(1.6, uPx * (0.0042 + 0.006 * a) / -mv.z);
          gl_Position = projectionMatrix * mv;
        }`,
      fragmentShader: /* glsl */`
        varying vec3 vC; varying float vA;
        void main(){
          float r = length(gl_PointCoord - 0.5);
          float a = smoothstep(0.5, 0.3, r) * vA;
          if (a < 0.004) discard;
          gl_FragColor = vec4(vC, a);
        }`,
    }));
    this.points.frustumCulled = false;
    this.scene.add(this.points);
  }

  buildSparks() {
    const segs = MAX_SPARKS * SPARK_SEGS;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(segs * 6), 3));
    geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(segs * 8), 4));
    this.sparkLines = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false }));
    this.sparkLines.frustumCulled = false;
    this.scene.add(this.sparkLines);
    const hg = new THREE.BufferGeometry();
    hg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(MAX_SPARKS * 3), 3));
    hg.setAttribute('alpha', new THREE.BufferAttribute(new Float32Array(MAX_SPARKS), 1));
    this.sparkHeadU = { uPx: { value: 1 }, uCol: { value: BCOL.dn.clone() } };
    this.sparkHeads = new THREE.Points(hg, new THREE.ShaderMaterial({
      uniforms: this.sparkHeadU, transparent: true, depthWrite: false,
      vertexShader: /* glsl */`
        attribute float alpha; uniform float uPx; varying float vA;
        void main(){ vA = alpha; vec4 mv = modelViewMatrix * vec4(position,1.0); gl_PointSize = uPx * 0.012 / -mv.z; gl_Position = projectionMatrix * mv; }`,
      fragmentShader: /* glsl */`
        uniform vec3 uCol; varying float vA;
        void main(){ float r = length(gl_PointCoord - 0.5); float a = smoothstep(0.5, 0.35, r) * vA; if (a < 0.01) discard; gl_FragColor = vec4(uCol, a); }`,
    }));
    this.sparkHeads.frustumCulled = false;
    this.scene.add(this.sparkHeads);
  }

  // KC→MBON "memory fan": one hairline per strong synapse; it fades as the synapse weakens.
  buildMemoryFan() {
    const c = this.circuits, fan = [];
    for (const m of c.mbons.filter((m) => m.primary !== false)) {
      const bodies = m.idx.map((i) => ({ i, p: this.P(i) }));
      if (!bodies.length) continue;
      for (const o of Object.keys(c.odors)) {
        const e = c.kc_mbon[m.type]?.[o];
        if (!e) continue;
        const order = e.kc.map((k, j) => [k, e.w[j]]).sort((a, b) => b[1] - a[1]).slice(0, 70);
        const wmax = order[0]?.[1] || 1;
        for (const [k, w] of order) {
          const kp = this.P(k);
          let best = bodies[0];
          for (const b of bodies) if (b.p.distanceToSquared(kp) < best.p.distanceToSquared(kp)) best = b;
          fan.push({ k, mbon: m.type, odor: o, w: w / wmax, a: kp, b: best.p });
        }
      }
    }
    this.fan = fan;
    const geo = new THREE.BufferGeometry();
    const p = new Float32Array(fan.length * 6);
    fan.forEach((f, i) => p.set([f.a.x, f.a.y, f.a.z, f.b.x, f.b.y, f.b.z], i * 6));
    geo.setAttribute('position', new THREE.BufferAttribute(p, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(fan.length * 8), 4).setUsage(THREE.DynamicDrawUsage));
    this.fanLines = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false }));
    this.fanLines.frustumCulled = false;
    this.scene.add(this.fanLines);
  }

  computeViews() {
    const c = this.circuits;
    const box = new THREE.Box3();
    for (let i = 0; i < this.N; i += 7) box.expandByPoint(this.P(i));
    const size = box.getSize(new THREE.Vector3()), center = box.getCenter(new THREE.Vector3());
    const top = new THREE.Box3();
    for (let i = 0; i < this.N; i += 5) { const p = this.P(i); if (p.y > center.y + 0.2) top.expandByPoint(p); }
    const bt = top.getSize(new THREE.Vector3());
    const kcs = [...(c.odors.A?.kc_idx || []), ...(c.odors.B?.kc_idx || [])];
    const mb = [...kcs, ...c.mbons.flatMap((m) => m.idx)];
    const mbBox = new THREE.Box3();
    for (const i of mb) mbBox.expandByPoint(this.P(i));
    const ms = mbBox.getSize(new THREE.Vector3());
    // each view is a box to frame (w × h); the distance is solved per frame from the popup's aspect
    this.views = {
      cns: { center: center.clone(), w: size.x, h: size.y, elev: 0.05 },
      brain: { center: top.getCenter(new THREE.Vector3()), w: bt.x, h: bt.y, elev: 0.12 },
      mb: { center: mbBox.getCenter(new THREE.Vector3()), w: Math.max(ms.x, 0.5), h: Math.max(ms.y, 0.4), elev: 0.2 },
    };
    this.setView('cns', true);
  }

  setView(name, instant = false) {
    const v = this.views?.[name];
    if (!v) return;
    this.viewName = name;
    Object.assign(this.viewTarget, { w: v.w, h: v.h, elev: v.elev }); this.viewTarget.center.copy(v.center);
    if (instant) { Object.assign(this.view, { w: v.w, h: v.h, elev: v.elev, dist: this.fitDist(v.w, v.h) }); this.view.center.copy(v.center); }
  }

  // Distance at which a w × h box fills the canvas with a margin, whichever side is tighter.
  fitDist(w, h) {
    const t = Math.tan((this.camera.fov * Math.PI) / 360), aspect = this.camera.aspect || 1;
    return Math.max((h * 0.5) / t, (w * 0.5) / (t * aspect)) * 1.12;
  }

  buildLabels() {
    const c = this.circuits;
    const add = (key, text, sub, idx) => {
      if (!idx?.length) return;
      const el = document.createElement('div');
      el.className = 'nlabel';
      el.innerHTML = `${text}${sub ? ` <i>${sub}</i>` : ''}`;
      el.style.opacity = 0;
      this.labelsEl.appendChild(el);
      this.labels.push({ key, el, anchor: this.centroid(idx), idx: idx.slice(0, 40), level: 0 });
    };
    add('kcA', 'Kenyon cells', 'toxic smell', c.odors.A?.kc_idx);
    add('kcB', 'Kenyon cells', 'honey smell', c.odors.B?.kc_idx);
    for (const m of c.mbons.filter((m) => m.primary !== false)) add(`mbon:${m.type}`, m.type, m.role, m.idx);
    const dans = c.dans.filter((d) => d.primary !== false);
    add('dan:punishment', 'PPL1', 'pain dopamine', dans.filter((d) => d.signal === 'punishment').flatMap((d) => d.idx));
    add('dan:reward', 'PAM', 'sugar dopamine', dans.filter((d) => d.signal === 'reward').flatMap((d) => d.idx));
    for (const [k, a] of Object.entries(c.actions)) {
      const name = k === 'feed' ? 'Fdg' : [...new Set((a.dn_types || []).map((t) => t.replace(/_[a-z]$/, '')))].join('/');
      add(`act:${k}`, name, (a.label || k).toLowerCase(), a.dn_idx);
    }
  }

  // ---------------------------------------------------------------- activity API
  flash(idx, color, a = 1, decay = 3) {
    for (const i of idx || []) {
      const h = this.hot.get(i);
      if (!h || h.a < a) this.hot.set(i, { a, color, decay });
    }
  }

  hold(idx, color, level) { if (level > 0.01 && idx?.length) this.sustained.push({ idx, color, level }); }

  // per-neuron levels (e.g. each LC neuron's own firing rate)
  holdEach(idx, levels, color, gain = 1, min = 0.05) {
    const i2 = [], l2 = [];
    for (let k = 0; k < idx.length; k++) { const v = levels[k] * gain; if (v > min) { i2.push(idx[k]); l2.push(v); } }
    if (i2.length) this.sustained.push({ idx: i2, color, levels: l2 });
  }

  firePath(path, speed = 1.2, delay = 0) {
    if (!path || path.length < 2) return;
    const pts = path.map((i) => this.P(i));
    const lens = [0];
    for (let i = 1; i < pts.length; i++) lens.push(lens[i - 1] + pts[i].distanceTo(pts[i - 1]));
    if (this.sparks.length >= MAX_SPARKS) this.sparks.shift();
    this.sparks.push({ path, pts, lens, total: lens.at(-1) || 1, t: -delay, speed, fired: new Set() });
  }

  sampleSpark(s, d) {
    d = Math.max(0, Math.min(s.total, d));
    let j = 1;
    while (j < s.lens.length - 1 && s.lens[j] < d) j++;
    const a = s.lens[j - 1], b = s.lens[j];
    return s.pts[j - 1].clone().lerp(s.pts[j], (d - a) / Math.max(1e-6, b - a));
  }

  updateFan(mb, conc) {
    if (!this.fan) return;
    const col = this.fanLines.geometry.attributes.color.array;
    this.fan.forEach((f, i) => {
      const g = mb.gain[f.mbon]?.get(f.k) ?? 1;
      const act = conc[f.odor] || 0;
      const a = (0.025 + 0.3 * act) * g * (0.35 + 0.65 * f.w);
      const c = f.odor === 'B' ? BCOL.odorB : BCOL.odorA;
      col.set([c.r, c.g, c.b, a, c.r, c.g, c.b, a * 0.4], i * 8);
    });
    this.fanLines.geometry.attributes.color.needsUpdate = true;
  }

  // ---------------------------------------------------------------- frame
  update(dt) {
    this.time += dt;
    const act = this.act, tint = this.tint;
    for (const i of this.touched) act[i] = 0;
    const touched = [];
    const put = (i, a, c) => {
      if (a <= act[i]) return;
      if (act[i] === 0) touched.push(i);
      act[i] = a; tint[i * 3] = c.r; tint[i * 3 + 1] = c.g; tint[i * 3 + 2] = c.b;
    };
    for (const s of this.sustained) {
      if (s.levels) for (let k = 0; k < s.idx.length; k++) put(s.idx[k], s.levels[k], s.color);
      else for (const i of s.idx) put(i, s.level, s.color);
    }
    for (const [i, h] of this.hot) {
      h.a *= Math.exp(-dt * h.decay);
      if (h.a < 0.02) { this.hot.delete(i); continue; }
      put(i, h.a, h.color);
    }
    this.touched = touched;
    this.sustained = [];
    if (!this.visible) return;                     // activity keeps integrating; drawing only when open
    this.points.geometry.attributes.act.needsUpdate = true;
    this.points.geometry.attributes.tint.needsUpdate = true;

    const lp = this.sparkLines.geometry.attributes.position.array, lc = this.sparkLines.geometry.attributes.color.array;
    const hp = this.sparkHeads.geometry.attributes.position.array, ha = this.sparkHeads.geometry.attributes.alpha.array;
    lp.fill(0); lc.fill(0); hp.fill(0); ha.fill(0);
    this.sparks = this.sparks.filter((s) => s.t < s.total + 0.5);
    const C = BCOL.dn;
    this.sparks.forEach((s, n) => {
      s.t += dt * s.speed;
      if (s.t < 0) return;
      const head = Math.min(s.t, s.total), tail = Math.max(0, s.t - 0.3);
      const fade = s.t > s.total ? Math.max(0, 1 - (s.t - s.total) / 0.5) : 1;
      for (let k = 0; k < SPARK_SEGS; k++) {
        const p0 = this.sampleSpark(s, tail + ((head - tail) * k) / SPARK_SEGS), p1 = this.sampleSpark(s, tail + ((head - tail) * (k + 1)) / SPARK_SEGS);
        const o = (n * SPARK_SEGS + k);
        lp.set([p0.x, p0.y, p0.z, p1.x, p1.y, p1.z], o * 6);
        lc.set([C.r, C.g, C.b, (k / SPARK_SEGS) * fade * 0.9, C.r, C.g, C.b, ((k + 1) / SPARK_SEGS) * fade * 0.9], o * 8);
      }
      const hpnt = this.sampleSpark(s, head);
      hp.set([hpnt.x, hpnt.y, hpnt.z], n * 3);
      ha[n] = fade;
      for (let j = 0; j < s.lens.length; j++) if (s.t >= s.lens[j] && !s.fired.has(j)) { s.fired.add(j); this.flash([s.path[j]], BCOL.dn, 1, 2); }
    });
    for (const a of ['position', 'color']) this.sparkLines.geometry.attributes[a].needsUpdate = true;
    this.sparkHeads.geometry.attributes.position.needsUpdate = true;
    this.sparkHeads.geometry.attributes.alpha.needsUpdate = true;

    const k = 1 - Math.exp(-dt * 1.8);
    this.view.center.lerp(this.viewTarget.center, k);
    this.view.dist += (this.fitDist(this.viewTarget.w, this.viewTarget.h) - this.view.dist) * k;
    this.view.elev += (this.viewTarget.elev - this.view.elev) * k;
    const orbit = Math.sin(this.time * 0.08) * 0.35;
    const d = this.view.dist, e = this.view.elev;
    this.camera.position.set(this.view.center.x + Math.sin(orbit) * d * Math.cos(e), this.view.center.y + Math.sin(e) * d, this.view.center.z + Math.cos(orbit) * d * Math.cos(e));
    this.camera.lookAt(this.view.center);
    const px = (this.renderer.getPixelRatio() * this.canvas.clientHeight * 0.5) / Math.tan((this.camera.fov * Math.PI) / 360);
    this.pointU.uPx.value = px;
    this.sparkHeadU.uPx.value = px;

    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    for (const l of this.labels) {
      let lv = 0;
      for (const i of l.idx) lv = Math.max(lv, act[i]);
      l.level = Math.max(l.level * Math.exp(-dt * 1.5), lv);
      const p = l.anchor.clone().project(this.camera);
      l.sx = ((p.x + 1) / 2) * w + 8; l.sy = ((1 - p.y) / 2) * h - 9; l.front = p.z < 1;
    }
    // show at most four labels, strongest first, skipping any that would overlap
    const placed = [];
    for (const l of [...this.labels].sort((a, b) => b.level - a.level)) {
      const ok = l.front && l.level > 0.35 && placed.length < 4 && placed.every((q) => Math.abs(q.sx - l.sx) > 110 || Math.abs(q.sy - l.sy) > 16);
      if (ok) placed.push(l);
      l.el.style.opacity = ok ? Math.min(1, (l.level - 0.35) * 2.5) : 0;
      l.el.style.transform = `translate(${l.sx}px, ${l.sy}px)`;
    }
  }

  setVisible(v) { this.visible = v; if (v) this.resize(); }

  resize() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  render() { if (this.visible) this.renderer.render(this.scene, this.camera); }
}
