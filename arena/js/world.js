// The road: an endless white studio floor with a lane. The robot fly walks
// forward forever; Jev steers it around (or toward) whatever gets dropped.
import * as THREE from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { Strider, ACCENT } from './robots.js';
import { Track, HORIZON, FOG_NEAR, FOG_FAR } from './track.js';

export function mulberry(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const ROAD = 0.62;          // half-width of the road
const FLY_R = 0.11;                // body collision radius
const LOOK = 1.7;                  // how far ahead obstacles are reported
const WALK = 0.36;
const BOT_SCALE = 0.24;             // Strider model units → world units
const SIGMA = 0.42;                // odor plume width (smell reaches ~1 unit)
const EDGE_PAIN = 0.8;             // hitting the border (touching toxic = 1.0)
const LANE_PAIN = 0.4;             // off-center pain just before the border; 0 on the center line
const EDGE_LIM = ROAD - 0.15;      // body centre where the outer feet meet the curb
const MAX_OBJ = 24;
const INK = new THREE.Color('#1c1c1c');

// Operant memory: what each motor program led to when the road was fully blocked.
// (The Pavlovian smell memory lives in the mushroom body; this one is action → outcome.)
export class MotorMemory {
  constructor() { this.reset(); }
  reset() { this.table = {}; this.attempt = null; }
  record(action, outcome) {
    const e = (this.table[action] ||= { tries: 0, hurt: 0, worked: 0, nothing: 0 });
    e.tries++; e[outcome]++;
  }
  summary() {
    const out = {};
    for (const a of ['walk_forward', 'veer_left', 'veer_right', 'walk_backward', 'takeoff']) {
      const e = this.table[a];
      out[a] = !e ? 'never tried' : e.worked ? `worked: got past it${e.worked > 1 ? ` (${e.worked} times)` : ''}` : e.hurt ? `hurt (${e.hurt} time${e.hurt > 1 ? 's' : ''})` : 'got nowhere';
    }
    return out;
  }
  get worked() { return Object.keys(this.table).filter((a) => this.table[a].worked); }
  get failed() { return Object.keys(this.table).filter((a) => !this.table[a].worked); }
}

export const KINDS = {
  toxic: { odor: 'A', solid: true, r: 0.15, label: 'toxic waste' },
  honey: { odor: 'B', solid: false, r: 0.19, label: 'honey' },
};

// Motor programs → body targets. w in rad/s, + = left. walk programs hold the road heading.
export const PROGRAMS = {
  walk_forward:  { v: WALK, align: 3.2 },
  veer_left:     { v: WALK * 0.9, target: 0.52, align: 3.5 },     // drift diagonally (~30°), not spin
  veer_right:    { v: WALK * 0.9, target: -0.52, align: 3.5 },
  walk_backward: { v: -0.22, align: 2 },
  takeoff:       { v: 0, mode: 'flight' },
  feed:          { v: 0, align: 1, mode: 'feed' },
};

function blobGeometry(seed, flat = 0.55, amp = 0.16) {
  const g = new THREE.SphereGeometry(1, 72, 44);
  const p = g.attributes.position, v = new THREE.Vector3(), r = mulberry(seed);
  const a = [r() * 6, r() * 6, r() * 6, r() * 6];
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i);
    const d = 1 + amp * Math.sin(3.1 * v.x + a[0]) * Math.sin(2.7 * v.z + a[1]) + amp * 0.5 * Math.sin(6.3 * v.y + a[2]) * Math.sin(5.1 * v.x + a[3]);
    v.multiplyScalar(d);
    if (v.y < 0) v.y *= 0.12;
    v.y *= flat;
    p.setXYZ(i, v.x, v.y, v.z);
  }
  g.computeVertexNormals();
  return g;
}

const TOXIC_MAT = new THREE.MeshPhysicalMaterial({ color: 0x151515, roughness: 0.32, metalness: 0.0, clearcoat: 0.8, clearcoatRoughness: 0.18 });
const HONEY_MAT = new THREE.MeshPhysicalMaterial({
  color: 0xf2b640, roughness: 0.04, transmission: 0.75, thickness: 0.12, ior: 1.5,
  clearcoat: 1, clearcoatRoughness: 0.02, attenuationColor: new THREE.Color('#b86a08'), attenuationDistance: 0.18,
});

export class World {
  constructor(canvas, labelsEl) {
    this.canvas = canvas;
    this.labelsEl = labelsEl;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.toneMapping = THREE.NeutralToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    this.scene = new THREE.Scene();
    this.scene.background = HORIZON.clone();
    this.scene.fog = new THREE.Fog(HORIZON.clone(), FOG_NEAR, FOG_FAR);
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environmentIntensity = 0.75;

    this.camera = new THREE.PerspectiveCamera(30, 1, 0.02, 60);
    this.camMode = 'chase';
    this.snapCam = true;
    this.camPos = new THREE.Vector3(1.3, 1.1, -1.6);
    this.camLook = new THREE.Vector3(0, 0, 0.9);

    this.time = 0;
    this.objects = [];
    this.nextId = 1;
    this.rand = mulberry(9);
    this.pain = 0;
    this.painAt = new THREE.Vector2();
    this.lastPainEvent = -9;
    this.lastEdgeEvent = -9;
    this.reward = 0;
    this.lanePain = 0;
    this.toxicPain = 0;              // the only pain that is paired with smells (Pavlovian US)
    this.motor = new MotorMemory();
    this.progress = { t: 0, z: 0 };   // odometry: when did the body last make real forward progress?
    this.stalled = 0;
    this.path = { blocked: false };
    this.road = ROAD;
    this.onEvent = () => {};

    this.buildLights();
    this.buildFloor();
    this.buildFx();
    this.buildTrail();

    this.bot = new Strider();
    this.bot.gait.freq = 3.4;          // quicker steps at road speed so feet stay planted
    this.bot.root.scale.setScalar(BOT_SCALE);
    this.bot.root.traverse((o) => { if (o.isMesh) o.castShadow = true; });
    this.scene.add(this.bot.root);
    this.feedAmt = 0;
    this.body = { x: 0, z: 0, heading: 0, v: 0, w: 0, alt: 0, program: 'walk_forward', programT: 0, arousal: 1, fed: 0, flight: null, contact: null, onHoney: null };

    this.resize();
    addEventListener('resize', () => this.resize());
  }

  // ---------------------------------------------------------------- scene
  buildLights() {
    this.scene.add(new THREE.HemisphereLight(0xeef1f5, 0xe2ddd3, 1.35));
    const key = new THREE.DirectionalLight(0xfff3e4, 2.5);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    const sc = key.shadow.camera;
    sc.left = -3; sc.right = 3; sc.top = 3; sc.bottom = -3; sc.near = 0.1; sc.far = 10;
    key.shadow.bias = -0.0004;
    key.shadow.normalBias = 0.01;
    this.key = key;
    this.scene.add(key, key.target);
  }

  buildFloor() {
    this.track = new Track(this.scene, { road: ROAD, maxObj: MAX_OBJ, sigma: SIGMA });
    this.floorU = this.track.floorU;
  }

  buildFx() {
    // contact shadow under the fly
    const c = document.createElement('canvas'); c.width = c.height = 128;
    const g = c.getContext('2d');
    const grad = g.createRadialGradient(64, 64, 0, 64, 64, 64);
    grad.addColorStop(0, 'rgba(0,0,0,0.42)'); grad.addColorStop(1, 'rgba(0,0,0,0)');
    g.fillStyle = grad; g.fillRect(0, 0, 128, 128);
    this.blob = new THREE.Mesh(new THREE.PlaneGeometry(0.34, 0.4), new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false }));
    this.blob.rotation.x = -Math.PI / 2;
    this.blob.position.y = 0.0015;
    this.scene.add(this.blob);

    // smoke puffs from the boosters
    const n = 360;
    this.smoke = { n, i: 0, pos: new Float32Array(n * 3), vel: new Float32Array(n * 3), life: new Float32Array(n), size: new Float32Array(n) };
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(this.smoke.pos, 3).setUsage(THREE.DynamicDrawUsage));
    geo.setAttribute('life', new THREE.BufferAttribute(this.smoke.life, 1).setUsage(THREE.DynamicDrawUsage));
    geo.setAttribute('size', new THREE.BufferAttribute(this.smoke.size, 1).setUsage(THREE.DynamicDrawUsage));
    this.smokeU = { uPx: { value: 1 } };
    this.smokePts = new THREE.Points(geo, new THREE.ShaderMaterial({
      uniforms: this.smokeU, transparent: true, depthWrite: false,
      vertexShader: /* glsl */`
        attribute float life; attribute float size; uniform float uPx; varying float vL;
        void main(){ vL = life; vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_PointSize = uPx * size / -mv.z; gl_Position = projectionMatrix * mv; }`,
      fragmentShader: /* glsl */`
        varying float vL;
        void main(){ float r = length(gl_PointCoord - 0.5); float a = smoothstep(0.5, 0.1, r) * vL * 0.22; if (a < 0.003) discard; gl_FragColor = vec4(vec3(0.62, 0.62, 0.6), a); }`,
    }));
    this.smokePts.frustumCulled = false;
    this.scene.add(this.smokePts);

    // blast / drop rings
    this.rings = [];
    this.ringGeo = new THREE.RingGeometry(0.96, 1, 96);
  }

  buildTrail() {
    this.trailN = 1400;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(this.trailN * 3), 3));
    geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(this.trailN * 4), 4));
    geo.setDrawRange(0, 0);
    this.trail = new THREE.Line(geo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false }));
    this.trail.frustumCulled = false;
    this.trailPts = [];
    this.trailClock = 0;
    this.scene.add(this.trail);
  }

  ring(x, z, { r0 = 0.05, r1 = 0.4, dur = 0.7, color = INK, opacity = 0.35 } = {}) {
    const m = new THREE.Mesh(this.ringGeo, new THREE.MeshBasicMaterial({ color, transparent: true, depthWrite: false, opacity }));
    m.rotation.x = -Math.PI / 2;
    m.position.set(x, 0.002, z);
    this.scene.add(m);
    this.rings.push({ m, t: 0, r0, r1, dur, opacity });
  }

  // ---------------------------------------------------------------- objects
  drop(kind, x, z, opts = {}) {
    const b = this.body;
    if (z === undefined) z = b.z + 2.4;
    if (kind === 'barrier') {
      const n = 5;
      for (let i = 0; i < n; i++) {
        const xx = -ROAD + 0.12 + (i * (2 * ROAD - 0.24)) / (n - 1);
        this.drop('toxic', xx + (this.rand() - 0.5) * 0.04, z + (this.rand() - 0.5) * 0.08, { r: 0.15 + this.rand() * 0.03, delay: i * 0.07, quiet: i > 0 });
      }
      this.onEvent({ type: 'drop', kind: 'barrier', x: 0, z });
      return;
    }
    const K = KINDS[kind];
    const r = opts.r || K.r * (0.9 + this.rand() * 0.2);
    x = Math.max(-ROAD + r * 0.6, Math.min(ROAD - r * 0.6, x ?? 0));
    const mesh = new THREE.Mesh(kind === 'toxic' ? blobGeometry(this.nextId * 13 + 1, 0.78, 0.14) : blobGeometry(this.nextId * 7 + 3, 0.22, 0.1), kind === 'toxic' ? TOXIC_MAT : HONEY_MAT);
    mesh.scale.setScalar(r);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.position.set(x, 1.4, z);
    this.scene.add(mesh);
    const o = { id: this.nextId++, kind, odor: K.odor, solid: K.solid, x, z, r, amount: 1, mesh, t: -(opts.delay || 0), landed: false, label: null };
    if (!opts.quiet) {
      const el = document.createElement('div');
      el.className = 'olabel';
      el.textContent = kind === 'toxic' ? 'toxic waste' : 'honey';
      this.labelsEl.appendChild(el);
      o.label = el;
    }
    this.objects.push(o);
    if (!opts.quiet) this.onEvent({ type: 'drop', kind, x, z });
    return o;
  }

  clearObjects() {
    for (const o of this.objects) { this.scene.remove(o.mesh); o.mesh.geometry.dispose(); o.label?.remove(); }
    this.objects = [];
  }

  // Screen point → road position (for click-to-drop).
  pick(clientX, clientY) {
    const r = this.canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    const ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, this.camera);
    const hit = new THREE.Vector3();
    if (!ray.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), hit)) return null;
    return { x: hit.x, z: hit.z };
  }

  // ---------------------------------------------------------------- motor
  setProgram(name, arousal = 1) {
    const b = this.body;
    if (b.flight || !PROGRAMS[name]) return false;
    b.arousal = arousal;
    if (name === 'takeoff') return this.takeoff();
    if (b.program !== name) b.programT = 0;
    b.program = name;
    this.bot.pulseLED();
    return true;
  }

  walled() { const p = this.path; return p.blocked && !p.left && !p.right; }

  takeoff() {
    const b = this.body;
    const M = this.motor;
    if (M.attempt) { M.record(M.attempt.action, M.attempt.hurt ? 'hurt' : 'nothing'); this.onEvent({ type: 'motor', action: M.attempt.action, outcome: M.attempt.hurt ? 'hurt' : 'nothing' }); M.attempt = null; }
    if (this.walled()) { const o = this.path.object; M.attempt = { action: 'takeoff', t: 0, hurt: false, passZ: o.z + o.r + FLY_R * 0.5, flying: true }; }
    // clear everything solid in the lane ahead
    let clear = b.z + 1.1;
    for (const o of this.objects) if (o.solid && o.z > b.z - 0.1 && o.z < b.z + LOOK + 0.5) clear = Math.max(clear, o.z + o.r + FLY_R + 0.35);
    const x1 = Math.max(-ROAD + 0.15, Math.min(ROAD - 0.15, b.x * 0.4));
    b.flight = { t: 0, dur: 0.9 + (clear - b.z) * 0.45, x0: b.x, z0: b.z, x1, z1: clear, h0: b.heading };
    b.program = 'takeoff';
    this.bot.pulseLED();
    this.ring(b.x, b.z, { r0: 0.06, r1: 0.45, dur: 0.8, opacity: 0.3 });
    this.onEvent({ type: 'takeoff' });
    return true;
  }

  resetTrail() { this.trailPts.length = 0; this.progress = { t: this.time, z: this.body.z }; }

  // ---------------------------------------------------------------- sensing
  concentration(x, z) {
    let A = 0, B = 0;
    for (const o of this.objects) {
      if (!o.landed) continue;
      const d2 = (x - o.x) ** 2 + (z - o.z) ** 2;
      const c = o.amount * Math.exp(-d2 / (2 * SIGMA * SIGMA));
      if (o.odor === 'A') A += c; else B += c;
    }
    return { A: Math.min(1, A), B: Math.min(1, B) };
  }

  egocentric(dx, dz) {
    const h = this.body.heading;
    const fwd = dx * Math.sin(h) + dz * Math.cos(h);
    const left = dx * Math.cos(h) - dz * Math.sin(h);
    const ang = Math.atan2(left, fwd), a = Math.abs(ang), side = ang > 0 ? 'left' : 'right';
    const where = a < 0.3 ? 'straight ahead' : a < 1.1 ? `ahead-${side}` : a < 2.1 ? `to the ${side}` : `behind-${side}`;
    return { where, dist: Math.hypot(dx, dz) };
  }

  // What blocks the lane ahead, and is there a way around it?
  pathAhead() {
    const b = this.body;
    const ahead = this.objects.filter((o) => o.landed && o.solid && o.z - b.z > -0.05 && o.z - b.z < LOOK);
    const reach = (o) => o.r * 0.95 + FLY_R;          // closest the body centre can get to a lump
    const inLane = ahead.filter((o) => Math.abs(o.x - b.x) < reach(o) + 0.03).sort((a, c) => a.z - c.z);
    if (!inLane.length) return { blocked: false, desc: 'clear road ahead' };
    const first = inLane[0];
    const row = ahead.filter((o) => Math.abs(o.z - first.z) < 0.35);
    const cover = row.map((o) => [o.x - reach(o), o.x + reach(o)]).sort((a, c) => a[0] - c[0]);
    const free = [];
    let cur = -EDGE_LIM;                             // the body centre can only reach ±EDGE_LIM
    for (const [lo, hi] of cover) { if (lo - cur > 0.02) free.push([cur, lo]); cur = Math.max(cur, hi); }
    if (EDGE_LIM - cur > 0.02) free.push([cur, EDGE_LIM]);
    // fly's left is +x when walking along +z
    const left = free.filter(([lo, hi]) => (lo + hi) / 2 > b.x).length > 0;
    const right = free.filter(([lo, hi]) => (lo + hi) / 2 <= b.x).length > 0;
    const dist = Math.max(0, first.z - b.z - first.r - FLY_R);
    return {
      blocked: true, dist, left, right, object: first,
      desc: dist < 0.05 ? 'solid lump right in front of you, touching' : `solid lump ${Math.max(1, Math.round(dist / 0.16))} body-lengths ahead, in your path`,
    };
  }

  sense() {
    const b = this.body, h = b.heading;
    const fx = Math.sin(h), fz = Math.cos(h), lx = Math.cos(h), lz = -Math.sin(h);
    const ant = (s) => this.concentration(b.x + fx * 0.09 + lx * 0.06 * s, b.z + fz * 0.09 + lz * 0.06 * s);
    const L = ant(1), Rr = ant(-1);
    let honeySeen = null;
    for (const o of this.objects) {
      if (!o.landed || o.kind !== 'honey' || o.amount < 0.05) continue;
      const dz = o.z - b.z;
      if (dz < -0.25 || dz > 2.6) continue;
      const e = this.egocentric(o.x - b.x, o.z - b.z);
      if (!honeySeen || e.dist < honeySeen.dist) honeySeen = e;
    }
    return {
      antennae: { A: { left: L.A, right: Rr.A }, B: { left: L.B, right: Rr.B } },
      path: this.path,
      puddle: honeySeen,
      pain: b.contact && b.edge ? `strong: wedged between a lump and the ${b.edge > 0 ? 'LEFT' : 'RIGHT'} road border, no room sideways: back away` : b.contact ? 'strong: pressing into a lump' : b.edge ? `strong: hitting the ${b.edge > 0 ? 'LEFT' : 'RIGHT'} road border` : null,
      lane: { x: b.x, reward: this.reward, pain: this.lanePain },
      walled: this.walled(),
      experience: Object.keys(this.motor.table).length || this.stalled > 2 ? this.motor.summary() : null,   // Jev judges from vision when it applies
      stalled: this.stalled,
      onSugar: !!b.onHoney,
      fed: b.fed,
      edgeLeft: EDGE_LIM - b.x < 0.08, edgeRight: b.x + EDGE_LIM < 0.08,
      heading: b.heading,
      airborne: !!b.flight,
      program: b.program, programT: b.programT,
    };
  }

  // ---------------------------------------------------------------- update
  update(dt) {
    this.time += dt;
    const t = this.time, b = this.body;
    b.programT += dt;
    b.contact = null; b.onHoney = null;

    if (b.flight) {
      const f = b.flight;
      f.t += dt / f.dur;
      const e = Math.min(1, f.t), s = e * e * (3 - 2 * e);
      b.x = f.x0 + (f.x1 - f.x0) * s;
      b.z = f.z0 + (f.z1 - f.z0) * s;
      b.alt = Math.pow(Math.sin(Math.PI * Math.min(1, e * 1.05)), 0.7) * 0.42;
      b.heading *= 1 - Math.min(1, dt * 3);
      b.v = 0; b.w = 0;
      this.boost = e < 0.15 ? 1 : e > 0.8 ? 0.9 : 0.55;
      if (f.t >= 1) {
        b.flight = null; b.alt = 0; b.program = 'walk_forward'; b.programT = 0; this.boost = 0;
        this.ring(b.x, b.z, { r0: 0.08, r1: 0.4, dur: 0.7, opacity: 0.25 });
        this.trailPts.push({ x: b.x, z: b.z, gap: true });
        this.onEvent({ type: 'land' });
      }
    } else {
      this.boost = 0;
      const P = PROGRAMS[b.program] || PROGRAMS.walk_forward;
      const gain = 0.85 + 0.1 * b.arousal;
      const wt = P.w !== undefined ? P.w : ((P.target || 0) - b.heading) * (P.align || 0);
      b.v += (P.v * gain - b.v) * (1 - Math.exp(-dt * 7));
      b.w += (wt - b.w) * (1 - Math.exp(-dt * 10));
      b.heading = Math.max(-1.05, Math.min(1.05, b.heading + b.w * dt));
      let nx = b.x + Math.sin(b.heading) * b.v * dt, nz = b.z + Math.cos(b.heading) * b.v * dt;
      const lim = EDGE_LIM, clampX = (x) => Math.max(-lim, Math.min(lim, x));
      const solids = this.objects.filter((o) => o.landed && o.solid);
      for (let it = 0; it < 2; it++) {
        nx = clampX(nx);
        for (const o of solids) {
          const dx = nx - o.x, dz = nz - o.z, d = Math.hypot(dx, dz), min = o.r * 0.95 + FLY_R;
          if (d < min + 0.012) b.contact = o;
          if (d < min) {
            // a head-on push deflects a little sideways (toward the heading), so the body slides off round lumps
            let ux = dx / (d || 1), uz = dz / (d || 1);
            if (Math.abs(ux) < 0.2) { ux += (b.heading >= 0 ? 0.25 : -0.25); const m = Math.hypot(ux, uz); ux /= m; uz /= m; }
            nx = o.x + ux * min; nz = o.z + uz * min;
          }
        }
      }
      nx = clampX(nx);
      // the curb won: if a lump still overlaps, the body is pushed back along the road instead
      b.wedged = false;
      for (const o of solids) {
        const dx = nx - o.x, dz = nz - o.z, min = o.r * 0.95 + FLY_R;
        if (dx * dx + dz * dz < min * min) {
          const along = Math.sqrt(Math.max(0, min * min - dx * dx));
          nz = o.z + (dz >= 0 ? along : -along);
          b.contact = o; b.wedged = true;
        }
      }
      b.edge = nx >= lim - 1e-4 ? 1 : nx <= -lim + 1e-4 ? -1 : 0;
      for (const o of this.objects) {
        if (!o.landed || o.solid) continue;
        if (Math.hypot(nx - o.x, nz - o.z) < o.r * Math.sqrt(o.amount) + FLY_R * 0.6 && o.amount > 0.05) b.onHoney = o;
      }
      b.x = nx; b.z = nz; b.alt = 0;
    }

    // pain from toxic contact
    this.toxicPain = b.contact ? 1 : Math.max(0, this.toxicPain - dt * 1.6);
    if (b.contact) {
      this.pain = 1;
      this.painAt.set(b.contact.x, b.contact.z);
      if (t - this.lastPainEvent > 0.9) { this.lastPainEvent = t; this.onEvent({ type: 'pain', object: b.contact }); this.bot.pulseLED(); }
    } else if (b.edge && !b.flight) {
      this.pain = Math.max(this.pain - dt * 1.6, EDGE_PAIN);
      this.painAt.set(b.edge * ROAD, b.z);
      if (t - this.lastEdgeEvent > 1.2) { this.lastEdgeEvent = t; this.track.flashCurb(b.edge, b.z); this.onEvent({ type: 'edge', side: b.edge }); this.bot.pulseLED(); }
    } else this.pain = Math.max(0, this.pain - dt * 1.6);

    // center line reward: maximal on the line, zero 0.3 away
    this.reward = b.flight ? 0 : Math.max(0, 1 - Math.abs(b.x) / 0.3);
    // off-center pain grows linearly with distance from the line (not fed to the mushroom body)
    this.lanePain = b.flight ? 0 : LANE_PAIN * Math.min(1, Math.abs(b.x) / EDGE_LIM);

    // odometry: seconds since the body last advanced 0.25 along the road
    if (b.flight || b.z > this.progress.z + 0.25) this.progress = { t, z: b.z };
    this.stalled = t - this.progress.t;

    // operant bookkeeping while the road is fully blocked
    if (!b.flight) this.path = this.pathAhead();
    const M = this.motor;
    if (M.attempt) {
      const A = M.attempt;
      A.t += dt;
      if (this.pain > 0.3) A.hurt = true;
      if (b.z > A.passZ) { M.record(A.action, 'worked'); this.onEvent({ type: 'motor', action: A.action, outcome: 'worked' }); M.attempt = null; }
      else if (!A.flying && A.t > 0.35 && (b.program !== A.action || !this.walled())) {
        const out = A.hurt ? 'hurt' : 'nothing';
        M.record(A.action, out); this.onEvent({ type: 'motor', action: A.action, outcome: out }); M.attempt = null;
      } else if (A.flying && !b.flight && A.t > 0.5) { M.record(A.action, A.hurt ? 'hurt' : 'nothing'); M.attempt = null; }
    }
    if (!M.attempt && !b.flight && this.walled() && this.path.dist < 1.4) {
      const o = this.path.object;
      M.attempt = { action: b.program, t: 0, hurt: this.pain > 0.3, passZ: o.z + o.r + FLY_R * 0.5, flying: false };
    }

    // feeding
    if (b.onHoney && b.program === 'feed') {
      b.onHoney.amount = Math.max(0, b.onHoney.amount - dt * 0.09);
      b.fed = Math.min(1, b.fed + dt / 3.5);
    }
    b.fed = Math.max(0, b.fed - dt / 30);

    // fly model
    const P = PROGRAMS[b.program] || {};
    this.feedAmt += ((P.mode === 'feed' && !b.flight ? 1 : 0) - this.feedAmt) * (1 - Math.exp(-dt * 5));
    const f = b.flight;
    const tuck = f ? (f.t < 0.5 ? Math.min(1, f.t / 0.15) : Math.min(1, b.alt / 0.05)) : 0;
    this.bot.root.position.set(b.x, 0, b.z);
    this.bot.root.rotation.y = b.heading;
    this.bot.update(dt, {
      v: b.v / BOT_SCALE, w: b.w, feed: this.feedAmt, flight: tuck, boost: this.boost || 0,
      alt: b.alt / BOT_SCALE, hover: 0, act: Math.min(1, Math.abs(b.v) / WALK + Math.abs(b.w) * 0.3),
    });
    this.blob.position.set(b.x, 0.0015, b.z - 0.01);
    this.blob.material.opacity = 1 - Math.min(0.85, b.alt * 2.2);
    this.blob.scale.setScalar(1 + b.alt * 2.2);

    // objects: drop, squash, feed-shrink, labels, cleanup
    const W = this.canvas.clientWidth, H = this.canvas.clientHeight;
    for (const o of this.objects) {
      o.t += dt;
      if (o.t < 0) { o.mesh.visible = false; continue; }
      o.mesh.visible = true;
      const fall = Math.min(1, o.t / 0.42);
      if (!o.landed && fall >= 1) { o.landed = true; this.ring(o.x, o.z, { r0: o.r * 0.8, r1: o.r * 2.6, dur: 0.6, opacity: 0.28 }); }
      const y = o.landed ? 0 : 1.4 * (1 - fall * fall);
      const settle = o.landed ? Math.exp(-(o.t - 0.42) * 7) * Math.sin((o.t - 0.42) * 26) * 0.18 : 0;
      const sz = o.r * (o.kind === 'honey' ? Math.sqrt(Math.max(0.05, o.amount)) : 1);
      o.mesh.position.set(o.x, y, o.z);
      o.mesh.scale.set(sz * (1 + settle * 0.5), sz * (1 - settle), sz * (1 + settle * 0.5));
      if (o.label) {
        const p = new THREE.Vector3(o.x, 0.12, o.z).project(this.camera);
        const age = o.t;
        o.label.style.opacity = age < 0.3 ? 0 : age < 2.6 ? 1 : Math.max(0, 1 - (age - 2.6) / 0.8);
        o.label.style.transform = `translate(-50%, -100%) translate(${((p.x + 1) / 2) * W}px, ${((1 - p.y) / 2) * H}px)`;
      }
    }
    this.objects = this.objects.filter((o) => {
      if (o.z > b.z - 3 && !(o.kind === 'honey' && o.amount <= 0.02)) return true;
      this.scene.remove(o.mesh); o.mesh.geometry.dispose(); o.label?.remove(); return false;
    });

    // rings
    this.rings = this.rings.filter((r) => {
      r.t += dt / r.dur;
      const e = 1 - Math.pow(1 - Math.min(1, r.t), 3);
      r.m.scale.setScalar(r.r0 + (r.r1 - r.r0) * e);
      r.m.material.opacity = r.opacity * (1 - Math.min(1, r.t));
      if (r.t < 1) return true;
      this.scene.remove(r.m); r.m.material.dispose(); return false;
    });

    // smoke from boosters
    const S = this.smoke;
    if (this.boost > 0.05) {
      const wp = new THREE.Vector3();
      for (const B of this.bot.boosters) {
        if (this.rand() > this.boost * 0.9) continue;
        B.g.getWorldPosition(wp);
        const i = S.i++ % S.n;
        S.pos.set([wp.x, wp.y - 0.03, wp.z], i * 3);
        S.vel.set([(this.rand() - 0.5) * 0.2, -0.9 - this.rand() * 0.4, (this.rand() - 0.5) * 0.2], i * 3);
        S.life[i] = 1; S.size[i] = 0.02;
      }
    }
    for (let i = 0; i < S.n; i++) {
      if (S.life[i] <= 0) continue;
      const k = i * 3;
      S.pos[k] += S.vel[k] * dt; S.pos[k + 1] += S.vel[k + 1] * dt; S.pos[k + 2] += S.vel[k + 2] * dt;
      if (S.pos[k + 1] < 0.01) {                          // hit the floor: spread out
        S.pos[k + 1] = 0.01;
        const sp = Math.hypot(S.vel[k], S.vel[k + 2]) || 1;
        const speed = Math.abs(S.vel[k + 1]) * 0.6 + sp;
        S.vel[k] = (S.vel[k] / sp) * speed; S.vel[k + 2] = (S.vel[k + 2] / sp) * speed; S.vel[k + 1] = 0.05;
      }
      S.vel[k] *= 1 - dt * 2.2; S.vel[k + 1] *= 1 - dt * 2.2; S.vel[k + 2] *= 1 - dt * 2.2;
      S.life[i] -= dt / 1.4;
      S.size[i] += dt * 0.09;
    }
    const sg = this.smokePts.geometry.attributes;
    sg.position.needsUpdate = true; sg.life.needsUpdate = true; sg.size.needsUpdate = true;

    // trail (a plotter line on the floor)
    this.trailClock -= dt;
    if (this.trailClock <= 0 && !b.flight) {
      this.trailClock = 0.04;
      this.trailPts.push({ x: b.x, z: b.z });
      if (this.trailPts.length > this.trailN) this.trailPts.shift();
    }
    const tp = this.trail.geometry.attributes.position.array, tc = this.trail.geometry.attributes.color.array;
    const n = this.trailPts.length;
    for (let i = 0; i < n; i++) {
      const p = this.trailPts[i], age = (i + 1) / n;
      tp[i * 3] = p.x; tp[i * 3 + 1] = 0.003; tp[i * 3 + 2] = p.z;
      tc[i * 4] = INK.r; tc[i * 4 + 1] = INK.g; tc[i * 4 + 2] = INK.b; tc[i * 4 + 3] = p.gap ? 0 : 0.05 + 0.4 * age;
    }
    this.trail.geometry.setDrawRange(0, n);
    this.trail.geometry.attributes.position.needsUpdate = true;
    this.trail.geometry.attributes.color.needsUpdate = true;

    // floor uniforms + follow
    const U = this.floorU;
    U.uTime.value = t;
    U.uFly.value.set(b.x, b.z);
    const landed = this.objects.filter((o) => o.landed).slice(-MAX_OBJ);
    landed.forEach((o, i) => U.uObj.value[i].set(o.x, o.z, o.odor === 'A' ? 1 : 2, o.amount));
    U.uN.value = landed.length;
    U.uPain.value = this.pain;
    U.uPainAt.value.copy(this.painAt);
    this.key.position.set(b.x + 2.3, 2.2, b.z + 1.9);
    this.key.target.position.set(b.x, 0, b.z + 0.6);

    // camera
    const fwd = b.z;
    let pos, look;
    if (this.camMode === 'hero') {
      pos = new THREE.Vector3(b.x + 0.5, 0.36 + b.alt * 0.8, fwd - 0.62);
      look = new THREE.Vector3(b.x, 0.07 + b.alt * 0.9, fwd + 0.12);
    } else if (this.camMode === 'side') {
      pos = new THREE.Vector3(2.4, 0.9 + b.alt * 0.4, fwd + 0.5);
      look = new THREE.Vector3(0, 0.05 + b.alt * 0.4, fwd + 0.7);
    } else {
      pos = new THREE.Vector3(b.x * 0.55 + 0.3, 1.25 + b.alt * 0.5, fwd - 2.0);
      look = new THREE.Vector3(b.x * 0.5, 0.0 + b.alt * 0.4, fwd + 0.55);
    }
    const kc = this.snapCam ? 1 : 1 - Math.exp(-dt * (this.camMode === 'hero' ? 3.2 : 2.2));
    this.snapCam = false;
    this.camPos.lerp(pos, kc);
    this.camLook.lerp(look, kc);
    this.camera.position.copy(this.camPos);
    this.camera.lookAt(this.camLook);
    this.smokeU.uPx.value = (this.renderer.getPixelRatio() * H * 0.5) / Math.tan((this.camera.fov * Math.PI) / 360);
    this.track.update(dt, b, this.reward, this.camera);
  }

  setCamera(mode) { this.camMode = mode; }

  resize() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  render() { this.renderer.render(this.scene, this.camera); }
}
