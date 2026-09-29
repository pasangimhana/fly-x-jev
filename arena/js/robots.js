// Four robot bodies for the same brain. All share one motor interface
// (v, w, feed, flight, boost, alt) and planted-foot IK legs.
import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';

export const ACCENT = new THREE.Color('#E55B2B');
const Y = new THREE.Vector3(0, 1, 0);
const TAU = Math.PI * 2;

export const MAT = {
  white: new THREE.MeshPhysicalMaterial({ color: 0xf3f3f0, roughness: 0.3, clearcoat: 0.6, clearcoatRoughness: 0.18 }),
  white2: new THREE.MeshPhysicalMaterial({ color: 0xe3e3de, roughness: 0.5, clearcoat: 0.25 }),
  graphite: new THREE.MeshStandardMaterial({ color: 0x1d1d1d, roughness: 0.42, metalness: 0.35 }),
  gloss: new THREE.MeshPhysicalMaterial({ color: 0x151515, roughness: 0.1, metalness: 0.25, clearcoat: 1, clearcoatRoughness: 0.04 }),
  steel: new THREE.MeshStandardMaterial({ color: 0xbdbdb8, roughness: 0.2, metalness: 1 }),
  rubber: new THREE.MeshStandardMaterial({ color: 0x2a2a2a, roughness: 0.9 }),
  glass: new THREE.MeshPhysicalMaterial({ color: 0x070707, roughness: 0.02, metalness: 0.1, clearcoat: 1, clearcoatRoughness: 0.02 }),
  accent: new THREE.MeshBasicMaterial({ color: ACCENT }),
  frost: new THREE.MeshPhysicalMaterial({ color: 0x2a2a2a, roughness: 0.15, transparent: true, opacity: 0.38, clearcoat: 1, side: THREE.DoubleSide, depthWrite: false }),
};

// ------------------------------------------------------------------ helpers
function mesh(geo, mat, parent) {
  const m = new THREE.Mesh(geo, mat);
  m.castShadow = true; m.receiveShadow = true;
  parent?.add(m);
  return m;
}
const box = (w, h, d, r, mat, parent) => mesh(new RoundedBoxGeometry(w, h, d, 3, r), mat, parent);
const sphere = (r, mat, parent, seg = 32) => mesh(new THREE.SphereGeometry(r, seg, Math.round(seg * 0.66)), mat, parent);

// Unit rod along +Y from origin; stretched between two points every frame.
function rod(r0, r1, mat, parent, seg = 6) {
  const g = new THREE.CylinderGeometry(r1, r0, 1, seg, 1);
  g.translate(0, 0.5, 0);
  return mesh(g, mat, parent);
}
const _d = new THREE.Vector3();
function setRod(m, a, b) {
  _d.subVectors(b, a);
  const len = _d.length() || 1e-6;
  m.position.copy(a);
  m.scale.set(1, len, 1);
  m.quaternion.setFromUnitVectors(Y, _d.divideScalar(len));
}
function axle(r, h, mat, parent, seg = 20) { return mesh(new THREE.CylinderGeometry(r, r, h, seg), mat, parent); }
function orient(m, axis) { m.quaternion.setFromUnitVectors(Y, axis.clone().normalize()); }

// Exhaust plume: white-hot at the nozzle, cooling to accent, fading out.
function plumeMat() {
  return new THREE.ShaderMaterial({
    transparent: true, depthWrite: false,
    uniforms: { uT: { value: 0 }, uP: { value: 0 }, uA: { value: ACCENT.clone() } },
    vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
    fragmentShader: `uniform float uT, uP; uniform vec3 uA; varying vec2 vUv;
      void main(){
        float along = vUv.y;
        float flick = 0.86 + 0.14 * sin(uT * 80.0 + vUv.x * 18.0);
        vec3 col = mix(vec3(1.0), uA, smoothstep(0.08, 0.55, along));
        float a = pow(1.0 - along, 1.6) * uP * flick;
        gl_FragColor = vec4(col, a * 0.85);
      }`,
  });
}

class Booster {
  constructor(parent, pos, r, dir = new THREE.Vector3(0, -1, 0), len = 9) {
    this.g = new THREE.Group();
    this.g.position.copy(pos);
    this.g.quaternion.setFromUnitVectors(new THREE.Vector3(0, -1, 0), dir.clone().normalize());
    parent.add(this.g);
    const shell = mesh(new THREE.CylinderGeometry(r * 0.82, r, r * 1.2, 24, 1, true), MAT.white2, this.g);
    shell.material = MAT.white2.clone(); shell.material.side = THREE.DoubleSide;
    const throat = mesh(new THREE.CircleGeometry(r * 0.8, 24), MAT.graphite, this.g);
    throat.rotation.x = Math.PI / 2; throat.position.y = -r * 0.45;
    this.glow = new THREE.Mesh(new THREE.CircleGeometry(r * 0.62, 24), new THREE.MeshBasicMaterial({ color: ACCENT, transparent: true, opacity: 0 }));
    this.glow.rotation.x = Math.PI / 2; this.glow.position.y = -r * 0.47;
    this.g.add(this.glow);
    const pg = new THREE.ConeGeometry(r * 0.75, r * len, 20, 1, true);
    pg.rotateX(Math.PI);
    pg.translate(0, -r * len * 0.5 - r * 0.45, 0);
    this.pm = plumeMat();
    this.plume = new THREE.Mesh(pg, this.pm);
    this.plume.renderOrder = 5;
    this.g.add(this.plume);
  }
  set(p, t) {
    const q = p * (0.92 + 0.08 * Math.sin(t * 60));
    this.pm.uniforms.uT.value = t; this.pm.uniforms.uP.value = q;
    this.plume.visible = q > 0.01;
    this.plume.scale.set(1, 0.55 + 0.45 * q, 1);
    this.glow.material.opacity = Math.min(1, q * 1.5);
  }
}

// ------------------------------------------------------------------ legs
// Two-link leg with analytic IK; the knee bends upward, out of the hip→foot line.
class Leg {
  constructor(body, o) {
    Object.assign(this, { hip: o.hip.clone(), rest: o.rest.clone(), L1: o.L1, L2: o.L2, offset: o.offset, tuck: o.tuck.clone(), side: Math.sign(o.hip.x) || 1 });
    this.ground = this.rest.clone();
    this.knee = new THREE.Vector3();
    const s = o.style;
    this.parts = {
      femur: rod(s.femurR, s.femurR * 0.85, s.femurMat, body, s.seg || 6),
      tibia: rod(s.tibiaR0, s.tibiaR1, s.tibiaMat, body, s.seg || 6),
      hipJ: s.hipJoint ? s.hipJoint(body) : null,
      kneeJ: s.kneeJoint ? s.kneeJoint(body) : null,
      foot: s.foot ? s.foot(body) : null,
      piston: s.piston ? rod(0.009, 0.009, MAT.steel, body, 8) : null,
      sleeve: s.piston ? rod(0.017, 0.017, MAT.graphite, body, 10) : null,
    };
    this.style = s;
  }

  pose(F) {
    const H = this.hip;
    const hf = new THREE.Vector3().subVectors(F, H);
    let d = hf.length();
    const maxd = this.L1 + this.L2 - 1e-3, mind = Math.abs(this.L1 - this.L2) + 1e-3;
    const dc = Math.min(maxd, Math.max(mind, d));
    const e1 = hf.clone().divideScalar(d || 1);
    const e2 = Y.clone().addScaledVector(e1, -Y.dot(e1));
    if (e2.lengthSq() < 1e-6) e2.set(this.side, 0, 0); else e2.normalize();
    const a = Math.acos(Math.min(1, Math.max(-1, (this.L1 * this.L1 + dc * dc - this.L2 * this.L2) / (2 * this.L1 * dc))));
    this.knee.copy(H).addScaledVector(e1, Math.cos(a) * this.L1).addScaledVector(e2, Math.sin(a) * this.L1);
    const foot = H.clone().addScaledVector(e1, dc);
    const p = this.parts;
    setRod(p.femur, H, this.knee);
    setRod(p.tibia, this.knee, foot);
    const n = new THREE.Vector3().crossVectors(e1, e2).normalize();
    if (p.hipJ) { p.hipJ.position.copy(H); if (this.style.hipAxis === 'plane') orient(p.hipJ, n); }
    if (p.kneeJ) { p.kneeJ.position.copy(this.knee); orient(p.kneeJ, n); }
    if (p.foot) p.foot.position.copy(foot);
    if (p.piston) {
      const base = H.clone().addScaledVector(Y, 0.07);
      const tip = H.clone().lerp(this.knee, 0.72).addScaledVector(e2, 0.02);
      const mid = base.clone().lerp(tip, 0.55);
      setRod(p.sleeve, base, mid);
      setRod(p.piston, mid.clone().lerp(base, 0.1), tip);
    }
  }
}

// Planted-foot gait: stance feet move with the ground, swing feet arc forward.
class Gait {
  constructor(legs, { freq = 2, duty = 0.55, lift = 0.07 } = {}) {
    Object.assign(this, { legs, freq, duty, lift });
    this.phase = 0;
  }
  update(dt, v, w, act) {
    const f = this.freq * (0.4 + 0.6 * act);
    if (act > 0.01) this.phase += dt * f;
    const Ts = this.duty / f;
    for (const L of this.legs) {
      const r = L.rest;
      const gx = -w * r.z, gz = -v + w * r.x;           // ground velocity under this foot (body frame)
      const s = ((this.phase + L.offset) % 1 + 1) % 1;
      let k, lift = 0;
      if (s < this.duty) k = s / this.duty - 0.5;
      else { const u = (s - this.duty) / (1 - this.duty); k = 0.5 - u; lift = Math.sin(Math.PI * u) * this.lift * Math.min(1, act * 2.5); }
      L.ground.set(r.x + gx * Ts * k, lift, r.z + gz * Ts * k);
    }
  }
}

// ------------------------------------------------------------------ base
class Bot {
  constructor() {
    this.root = new THREE.Group();
    this.body = new THREE.Group();
    this.root.add(this.body);
    this.legs = []; this.boosters = [];
    this.t = 0; this.bodyY = 0;
    this.ride = 0.24; this.feedDrop = 0.06; this.vmax = 0.5;
  }
  addLeg(o) { const L = new Leg(this.body, o); this.legs.push(L); return L; }

  // st: { v, w, feed, flight, boost, alt, act }
  update(dt, st) {
    this.t += dt;
    this.bodyY = this.ride - st.feed * this.feedDrop + st.alt;
    this.body.position.y = this.bodyY;
    this.gait.update(dt, st.v, st.w, st.act * (1 - st.flight));
    const tk = st.flight * st.flight * (3 - 2 * st.flight);
    for (const L of this.legs) {
      const g = L.ground.clone(); g.y -= this.bodyY;
      L.pose(g.lerp(L.tuck, tk));
    }
    for (const B of this.boosters) B.set(st.boost, this.t);
    this.animate(dt, st);
  }
  animate() {}
  // Where the feeder tip touches the ground (body-relative x/z) for placing food.
  get mouth() { return new THREE.Vector3(0, 0, 0.45); }
}

// ================================================================== 01 STRIDER
export class Strider extends Bot {
  constructor() {
    super();
    this.name = 'Strider';
    this.ride = 0.2; this.feedDrop = 0.07; this.vmax = 0.5;
    const B = this.body;
    const tray = mesh(new THREE.CylinderGeometry(0.33, 0.29, 0.07, 6), MAT.graphite, B);
    tray.rotation.y = Math.PI / 6; tray.position.y = -0.03;
    const shell = mesh(new THREE.CylinderGeometry(0.3, 0.345, 0.1, 6), MAT.white, B);
    shell.rotation.y = Math.PI / 6; shell.position.y = 0.05;
    const belt = mesh(new THREE.CylinderGeometry(0.348, 0.348, 0.014, 6), MAT.accent, B);
    belt.rotation.y = Math.PI / 6; belt.position.y = 0.0;
    belt.scale.set(1, 1, 1);
    const hatch = box(0.28, 0.03, 0.2, 0.012, MAT.white2, B); hatch.position.set(0, 0.11, -0.03);
    for (const x of [-0.06, 0, 0.06]) { const slit = box(0.012, 0.006, 0.14, 0.003, MAT.graphite, B); slit.position.set(x, 0.127, -0.03); }
    this.led = sphere(0.012, MAT.accent, B, 12); this.led.position.set(0.1, 0.13, 0.06);
    const fin = box(0.012, 0.07, 0.08, 0.005, MAT.graphite, B); fin.position.set(-0.09, 0.15, -0.14);
    // head + lidar visor
    const head = box(0.27, 0.11, 0.13, 0.035, MAT.white, B); head.position.set(0, 0.04, 0.36);
    const visor = box(0.23, 0.04, 0.03, 0.012, MAT.glass, B); visor.position.set(0, 0.05, 0.425);
    this.scan = box(0.034, 0.012, 0.006, 0.003, MAT.accent, B); this.scan.position.set(0, 0.05, 0.442);
    for (const s of [1, -1]) { const cheek = axle(0.028, 0.03, MAT.graphite, B); cheek.rotation.z = Math.PI / 2; cheek.position.set(s * 0.15, 0.03, 0.37); }
    // feeder: telescoping probe
    this.probe = new THREE.Group(); this.probe.position.set(0, -0.02, 0.42); B.add(this.probe);
    this.probeSegs = [axle(0.02, 0.1, MAT.graphite, this.probe), axle(0.013, 0.1, MAT.steel, this.probe), axle(0.008, 0.08, MAT.accent, this.probe)];
    // lift jets
    for (const [x, z] of [[0.19, 0.15], [-0.19, 0.15], [0.19, -0.15], [-0.19, -0.15]]) this.boosters.push(new Booster(B, new THREE.Vector3(x, -0.07, z), 0.04));
    // legs at the six vertices
    const style = {
      femurR: 0.03, femurMat: MAT.white, tibiaR0: 0.024, tibiaR1: 0.011, tibiaMat: MAT.graphite, piston: true,
      hipJoint: (b) => axle(0.048, 0.075, MAT.graphite, b),
      kneeJoint: (b) => { const g = new THREE.Group(); b.add(g); axle(0.033, 0.07, MAT.graphite, g); axle(0.014, 0.078, MAT.steel, g); return g; },
      foot: (b) => { const f = sphere(0.026, MAT.rubber, b, 16); f.scale.set(1, 0.6, 1); return f; },
    };
    const angles = [30, 90, 150, 210, 270, 330];
    angles.forEach((deg, i) => {
      const a = (deg * Math.PI) / 180, dir = new THREE.Vector3(Math.sin(a), 0, Math.cos(a));
      const hip = dir.clone().multiplyScalar(0.33);
      const rest = dir.clone().multiplyScalar(0.68); rest.z += dir.z * 0.08; rest.y = 0;
      const tuck = dir.clone().multiplyScalar(0.36); tuck.y = -0.14;
      const offset = [0, 0.5, 0, 0.5, 0, 0.5][i];
      this.addLeg({ hip, rest, tuck, L1: 0.3, L2: 0.34, offset, style });
    });
    this.gait = new Gait(this.legs, { freq: 2.2, duty: 0.55, lift: 0.08 });
  }
  animate(dt, st) {
    this.scan.position.x = Math.sin(this.t * 3.2) * 0.085;
    const ext = st.feed;
    this.probe.rotation.x = -0.9;
    this.probeSegs[0].position.y = -0.05;
    this.probeSegs[1].position.y = -0.05 - 0.09 * ext;
    this.probeSegs[2].position.y = -0.05 - 0.17 * ext;
    this.probe.visible = ext > 0.02;
    this.flash = Math.max(0, (this.flash || 0) - dt * 4);
    this.led.material = this.flash > 0.1 || st.boost > 0.1 || (Math.floor(this.t * 2) % 2 && st.act > 0.1) ? MAT.accent : MAT.white2;
    this.body.rotation.x = -0.08 * st.flight + 0.05 * st.feed;
    this.body.rotation.z = -st.w * 0.05;
  }
  pulseLED() { this.flash = 1; }
  get mouth() { return new THREE.Vector3(0, 0, 0.56); }
}

// ================================================================== 02 HOVER
export class Hover extends Bot {
  constructor() {
    super();
    this.name = 'Hover';
    this.ride = 0.24; this.feedDrop = 0.07; this.vmax = 0.45;
    const B = this.body;
    const cap = new THREE.CapsuleGeometry(0.12, 0.3, 8, 32); cap.rotateX(Math.PI / 2);
    const hull = mesh(cap, MAT.white, B); hull.scale.set(1.05, 0.85, 1); hull.position.y = 0.04;
    const belly = mesh(cap.clone(), MAT.graphite, B); belly.scale.set(0.92, 0.7, 0.96); belly.position.y = 0.0;
    const spine = box(0.03, 0.02, 0.3, 0.008, MAT.graphite, B); spine.position.set(0, 0.145, -0.02);
    // single camera eye
    const eye = mesh(new THREE.SphereGeometry(0.075, 32, 20, 0, TAU, 0, Math.PI / 2), MAT.glass, B);
    eye.rotation.x = Math.PI / 2; eye.position.set(0, 0.05, 0.3);
    this.iris = mesh(new THREE.TorusGeometry(0.052, 0.006, 10, 48), MAT.accent, B); this.iris.position.set(0, 0.05, 0.352);
    const bezel = mesh(new THREE.TorusGeometry(0.078, 0.01, 10, 48), MAT.white2, B); bezel.position.set(0, 0.05, 0.3);
    // arms + ducted rotors
    this.rotors = [];
    const ductGeo = new THREE.TorusGeometry(0.12, 0.022, 14, 56); ductGeo.rotateX(Math.PI / 2);
    for (const [sx, sz] of [[1, 1], [-1, 1], [1, -1], [-1, -1]]) {
      const c = new THREE.Vector3(sx * 0.4, 0.07, sz * 0.3);
      const arm = rod(0.022, 0.018, MAT.white, B, 8);
      setRod(arm, new THREE.Vector3(sx * 0.09, 0.05, sz * 0.1), c.clone().add(new THREE.Vector3(-sx * 0.11, 0, -sz * 0.08)));
      const duct = mesh(ductGeo, MAT.white, B); duct.position.copy(c);
      const lip = mesh(new THREE.CylinderGeometry(0.118, 0.118, 0.035, 40, 1, true), MAT.graphite, B); lip.position.copy(c);
      lip.material = MAT.graphite.clone(); lip.material.side = THREE.DoubleSide;
      const hub = axle(0.022, 0.03, MAT.graphite, B); hub.position.copy(c);
      const rotor = new THREE.Group(); rotor.position.copy(c); B.add(rotor);
      for (let k = 0; k < 3; k++) { const bl = box(0.1, 0.004, 0.022, 0.002, MAT.graphite, rotor); bl.position.x = 0.055; const p = new THREE.Group(); p.rotation.y = (k * TAU) / 3; p.add(bl); rotor.add(p); bl.rotation.x = 0.25; }
      const blur = new THREE.Mesh(new THREE.CircleGeometry(0.11, 40), new THREE.MeshBasicMaterial({ color: 0x1c1c1c, transparent: true, opacity: 0, depthWrite: false }));
      blur.rotation.x = -Math.PI / 2; blur.position.copy(c).add(new THREE.Vector3(0, 0.003, 0)); B.add(blur);
      this.rotors.push({ rotor, blur, spin: 0, dir: sx * sz });
    }
    // four spider legs from the arm roots
    const style = {
      femurR: 0.016, femurMat: MAT.graphite, tibiaR0: 0.012, tibiaR1: 0.006, tibiaMat: MAT.graphite, seg: 8,
      hipJoint: (b) => sphere(0.026, MAT.white2, b, 16),
      kneeJoint: (b) => axle(0.02, 0.04, MAT.white, b),
      foot: (b) => sphere(0.014, MAT.accent, b, 12),
    };
    [[1, 1, 0], [-1, 1, 0.5], [1, -1, 0.5], [-1, -1, 0]].forEach(([sx, sz, off]) => {
      const hip = new THREE.Vector3(sx * 0.2, -0.01, sz * 0.18);
      const rest = new THREE.Vector3(sx * 0.46, 0, sz * 0.42);
      const tuck = new THREE.Vector3(sx * 0.3, -0.03, sz * 0.24);
      this.addLeg({ hip, rest, tuck, L1: 0.2, L2: 0.28, offset: off, style });
    });
    this.gait = new Gait(this.legs, { freq: 2.6, duty: 0.6, lift: 0.07 });
    this.straw = [axle(0.012, 0.09, MAT.graphite, B), axle(0.008, 0.09, MAT.accent, B)];
  }
  animate(dt, st) {
    for (const R of this.rotors) {
      const target = 3 + 55 * st.boost;
      R.spin += (target - R.spin) * (1 - Math.exp(-dt * 3));
      R.rotor.rotation.y += R.dir * R.spin * dt;
      R.blur.material.opacity = Math.min(0.22, Math.max(0, (R.spin - 15) / 120));
      R.rotor.visible = R.spin < 45;
    }
    this.iris.scale.setScalar(1 + 0.12 * Math.sin(this.t * 2));
    this.body.rotation.x = 0.12 * st.flight * (1 - st.hover) + 0.04 * st.feed;
    this.body.rotation.z = -st.w * 0.07 - 0.1 * st.flight * Math.sin(this.t * 1.3) * st.hover;
    const e = st.feed;
    this.straw.forEach((s, i) => { s.visible = e > 0.02; s.rotation.x = 2.04; s.position.set(0, -0.0 - 0.04 * i * e - 0.03, 0.32 + (0.04 + 0.07 * i) * e); });
  }
  get mouth() { return new THREE.Vector3(0, 0, 0.47); }
}

// ================================================================== 03 SCARAB
export class Scarab extends Bot {
  constructor() {
    super();
    this.name = 'Scarab';
    this.ride = 0.17; this.feedDrop = 0.05; this.vmax = 0.4;
    const B = this.body;
    const under = sphere(1, MAT.white, B, 40); under.scale.set(0.24, 0.1, 0.34); under.position.set(0, 0.02, -0.04);
    // two elytra halves hinged along the spine
    this.elytra = [];
    for (const s of [1, -1]) {
      const pivot = new THREE.Group(); pivot.position.set(0, 0.07, 0.16); B.add(pivot);
      const g = new THREE.SphereGeometry(1, 48, 28, s > 0 ? Math.PI / 2 : Math.PI * 1.5, Math.PI, 0, Math.PI / 2);
      const shell = mesh(g, MAT.gloss, pivot); shell.scale.set(0.27, 0.19, 0.4); shell.position.set(0, 0, -0.2);
      shell.material = MAT.gloss.clone(); shell.material.side = THREE.DoubleSide;
      const wing = new THREE.Mesh(new THREE.PlaneGeometry(0.16, 0.42), MAT.frost);
      const wp = new THREE.Group(); wp.position.set(s * 0.02, 0.05, 0.12); B.add(wp);
      wing.position.set(s * 0.08, 0, -0.2); wing.rotation.x = -Math.PI / 2; wp.add(wing);
      this.elytra.push({ pivot, side: s, wp });
    }
    const pronotum = box(0.34, 0.1, 0.14, 0.04, MAT.white2, B); pronotum.position.set(0, 0.07, 0.2);
    const head = box(0.2, 0.08, 0.12, 0.03, MAT.graphite, B); head.position.set(0, 0.05, 0.32);
    this.lenses = [];
    for (const s of [1, -1]) {
      const lens = sphere(0.04, MAT.glass, B, 24); lens.position.set(s * 0.065, 0.07, 0.37);
      const ring = mesh(new THREE.TorusGeometry(0.028, 0.005, 8, 32), MAT.accent, B); ring.position.set(s * 0.065, 0.07, 0.408);
      this.lenses.push(ring);
    }
    this.mandibles = [];
    for (const s of [1, -1]) {
      const p = new THREE.Group(); p.position.set(s * 0.05, 0.02, 0.38); B.add(p);
      const m = box(0.02, 0.018, 0.09, 0.006, MAT.gloss, p); m.position.set(0, 0, 0.04); m.rotation.y = -s * 0.3;
      this.mandibles.push({ p, s });
    }
    this.tongue = axle(0.01, 0.12, MAT.accent, B); this.tongue.rotation.x = 1.94;
    this.boosters.push(new Booster(B, new THREE.Vector3(0.09, 0.08, -0.22), 0.05, new THREE.Vector3(0, -1, -0.35)));
    this.boosters.push(new Booster(B, new THREE.Vector3(-0.09, 0.08, -0.22), 0.05, new THREE.Vector3(0, -1, -0.35)));
    const style = {
      femurR: 0.024, femurMat: MAT.white, tibiaR0: 0.02, tibiaR1: 0.01, tibiaMat: MAT.gloss,
      hipJoint: (b) => sphere(0.03, MAT.graphite, b, 16),
      kneeJoint: (b) => axle(0.024, 0.05, MAT.graphite, b),
      foot: (b) => { const f = mesh(new THREE.ConeGeometry(0.016, 0.04, 10), MAT.graphite, b); return f; },
    };
    const hips = [[0.14, 0.12], [0.17, 0.0], [0.15, -0.14]];
    hips.forEach(([x, z], i) => {
      for (const s of [1, -1]) {
        const hip = new THREE.Vector3(s * x, -0.01, z);
        const rest = new THREE.Vector3(s * (x + 0.28), 0, z * 1.8 + (i === 0 ? 0.08 : i === 2 ? -0.06 : 0));
        const tuck = new THREE.Vector3(s * (x + 0.06), -0.06, z);
        const offset = ((s === 1) === (i !== 1)) ? 0 : 0.5;
        this.addLeg({ hip, rest, tuck, L1: 0.17, L2: 0.23, offset, style });
      }
    });
    this.gait = new Gait(this.legs, { freq: 2.8, duty: 0.55, lift: 0.06 });
  }
  animate(dt, st) {
    const open = Math.min(1, st.flight * 1.4);
    const e = open * open * (3 - 2 * open);
    for (const E of this.elytra) {
      E.pivot.rotation.set(-0.3 * e, 0, E.side * 0.95 * e);
      E.pivot.position.set(E.side * 0.13 * e, 0.07 + 0.05 * e, 0.16);
      E.wp.rotation.set(0, -E.side * 1.0 * e, E.side * (0.2 * e + Math.sin(this.t * 40) * 0.1 * e * st.boost));
      E.wp.visible = e > 0.02;
    }
    for (const M of this.mandibles) M.p.rotation.y = M.s * (0.1 + 0.55 * st.feed + 0.1 * Math.sin(this.t * 9) * st.feed);
    this.tongue.visible = st.feed > 0.05;
    this.tongue.position.set(0, -0.0 - 0.02 * st.feed, 0.4 + 0.05 * st.feed);
    this.tongue.scale.y = 0.2 + 0.8 * st.feed;
        this.body.rotation.x = -0.18 * st.flight + 0.05 * st.feed;
    this.body.rotation.z = -st.w * 0.05;
  }
  get mouth() { return new THREE.Vector3(0, 0, 0.5); }
}

// ================================================================== 04 ORB
export class Orb extends Bot {
  constructor() {
    super();
    this.name = 'Orb';
    this.ride = 0.42; this.feedDrop = 0.17; this.vmax = 0.55;
    const B = this.body;
    this.shell = new THREE.Group(); B.add(this.shell);
    sphere(0.2, MAT.white, this.shell, 64);
    const band = mesh(new THREE.CylinderGeometry(0.203, 0.203, 0.05, 64, 1, true), MAT.graphite, this.shell);
    band.material = MAT.graphite.clone(); band.material.side = THREE.DoubleSide;
    const capTop = mesh(new THREE.SphereGeometry(0.06, 24, 12, 0, TAU, 0, Math.PI / 2), MAT.graphite, this.shell); capTop.position.y = 0.19;
    this.led = sphere(0.012, MAT.accent, this.shell, 12); this.led.position.set(0, 0.25, 0);
    // the eye: a gimbal that looks where it steers
    this.eye = new THREE.Group(); B.add(this.eye);
    const lens = mesh(new THREE.SphereGeometry(0.2015, 48, 24, 0, TAU, 0, 0.5), MAT.glass, this.eye); lens.rotation.x = Math.PI / 2;
    this.iris = mesh(new THREE.TorusGeometry(0.068, 0.008, 10, 64), MAT.accent, this.eye); this.iris.position.z = 0.19;
    const pupil = mesh(new THREE.TorusGeometry(0.03, 0.004, 8, 40), MAT.accent, this.eye); pupil.position.z = 0.197;
    this.needle = axle(0.006, 0.3, MAT.steel, this.eye); this.needle.rotation.x = Math.PI / 2 + 0.9;
    // ring thruster underneath
    const ring = mesh(new THREE.TorusGeometry(0.1, 0.025, 12, 48), MAT.white2, B); ring.rotation.x = Math.PI / 2; ring.position.y = -0.2;
    this.ringGlow = new THREE.Mesh(new THREE.TorusGeometry(0.1, 0.012, 10, 48), new THREE.MeshBasicMaterial({ color: ACCENT, transparent: true, opacity: 0 }));
    this.ringGlow.rotation.x = Math.PI / 2; this.ringGlow.position.y = -0.215; B.add(this.ringGlow);
    for (let k = 0; k < 6; k++) { const a = (k / 6) * TAU; this.boosters.push(new Booster(B, new THREE.Vector3(Math.cos(a) * 0.1, -0.21, Math.sin(a) * 0.1), 0.022, new THREE.Vector3(0, -1, 0), 12)); }
    // four harvestman legs: knees high over the body
    const style = {
      femurR: 0.014, femurMat: MAT.graphite, tibiaR0: 0.012, tibiaR1: 0.005, tibiaMat: MAT.graphite, seg: 8,
      hipJoint: (b) => sphere(0.03, MAT.white2, b, 16),
      kneeJoint: (b) => sphere(0.024, MAT.white, b, 16),
      foot: (b) => sphere(0.012, MAT.accent, b, 12),
    };
    [[45, 0], [135, 0.5], [225, 0], [315, 0.5]].forEach(([deg, off]) => {
      const a = (deg * Math.PI) / 180, dir = new THREE.Vector3(Math.sin(a), 0, Math.cos(a));
      const hip = dir.clone().multiplyScalar(0.19); hip.y = -0.02;
      const rest = dir.clone().multiplyScalar(0.72); rest.y = 0;
      const tuck = dir.clone().multiplyScalar(0.3); tuck.y = -0.12;
      this.addLeg({ hip, rest, tuck, L1: 0.46, L2: 0.62, offset: off, style });
    });
    this.gait = new Gait(this.legs, { freq: 1.35, duty: 0.62, lift: 0.1 });
    this.look = 0;
  }
  animate(dt, st) {
    this.look += (st.w * 0.55 - this.look) * (1 - Math.exp(-dt * 5));
    this.eye.rotation.set(0.35 * st.feed, this.look, 0);
    this.needle.visible = st.feed > 0.03;
    this.needle.scale.y = st.feed;
    this.needle.position.set(0, -0.05 - 0.07 * st.feed, 0.2 + 0.1 * st.feed);
    this.iris.scale.setScalar(1 + 0.1 * Math.sin(this.t * 2.2));
    this.ringGlow.material.opacity = Math.min(1, st.boost * 1.3);
    this.shell.rotation.y += dt * 1.2 * st.hover;
    this.led.visible = Math.sin(this.t * 3) > 0 || st.boost > 0.1;
    this.body.rotation.z = -st.w * 0.08;
    this.body.rotation.x = 0.05 * st.feed;
  }
  get mouth() { return new THREE.Vector3(0, 0, 0.5); }
}

export const BOTS = { Strider, Hover, Scarab, Orb };
