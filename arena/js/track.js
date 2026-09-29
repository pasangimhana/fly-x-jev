// The test track the robot walks: sky, concrete ground, a bright runway with
// physical curbs and light strips, painted distance numerals. The floor shader also draws smell fields, the rewarded center
// line and pain ripples.
import * as THREE from 'three';

export const HORIZON = new THREE.Color('#f2f0eb');
const SKY_TOP = new THREE.Color('#e1e4e7');
const GROUND = new THREE.Color('#e4e1da');
const RUNWAY = new THREE.Color('#fbfaf7');
const INK = new THREE.Color('#1c1c1c');
const ACCENT = new THREE.Color('#E55B2B');
const HONEY = new THREE.Color('#c98a1c');
export const FOG_NEAR = 4, FOG_FAR = 13;

const NUM_EVERY = 2, NUM_N = 9;
const CURB_W = 0.05, CURB_H = 0.035;

export class Track {
  constructor(scene, { road, maxObj, sigma }) {
    Object.assign(this, { scene, road, maxObj, sigma });
    this.buildSky();
    this.buildFloor();
    this.buildCurbs();
    this.buildNumerals();
  }

  buildSky() {
    const sky = new THREE.Mesh(new THREE.SphereGeometry(40, 32, 16), new THREE.ShaderMaterial({
      side: THREE.BackSide, depthWrite: false, fog: false,
      uniforms: { uTop: { value: SKY_TOP.clone() }, uHorizon: { value: HORIZON.clone() } },
      vertexShader: 'varying vec3 vD; void main(){ vD = normalize(position); vec4 p = projectionMatrix * modelViewMatrix * vec4(position, 1.0); gl_Position = p.xyww; }',
      fragmentShader: `uniform vec3 uTop, uHorizon; varying vec3 vD;
        void main(){ float h = clamp(vD.y, 0.0, 1.0); gl_FragColor = vec4(mix(uHorizon, uTop, pow(h, 0.55)), 1.0);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
        }`,
    }));
    sky.renderOrder = -10;
    sky.frustumCulled = false;
    this.sky = sky;
    this.scene.add(sky);
  }

  buildFloor() {
    const R = this.road.toFixed(3), N = this.maxObj;
    this.floorU = {
      uTime: { value: 0 }, uFly: { value: new THREE.Vector2() },
      uObj: { value: Array.from({ length: N }, () => new THREE.Vector4()) }, uN: { value: 0 },
      uPain: { value: 0 }, uPainAt: { value: new THREE.Vector2() }, uReward: { value: 0 },
      uInk: { value: INK.clone() }, uAccent: { value: ACCENT.clone() }, uHoney: { value: HONEY.clone() },
      uGround: { value: GROUND.clone() }, uRunway: { value: RUNWAY.clone() }, uHorizon: { value: HORIZON.clone() },
    };
    const mat = new THREE.ShaderMaterial({
      uniforms: this.floorU,
      vertexShader: /* glsl */`
        varying vec2 vW; varying float vDepth;
        void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xz; vec4 mv = viewMatrix * w; vDepth = -mv.z; gl_Position = projectionMatrix * mv; }`,
      fragmentShader: /* glsl */`
        uniform float uTime, uN, uPain, uReward; uniform vec2 uFly, uPainAt;
        uniform vec4 uObj[${N}];
        uniform vec3 uInk, uAccent, uHoney, uGround, uRunway, uHorizon;
        varying vec2 vW; varying float vDepth;
        float aa(float d, float w){ return 1.0 - smoothstep(-w, w, d); }
        float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
        float noise(vec2 p){ vec2 i = floor(p), f = fract(p); vec2 u = f*f*(3.0-2.0*f);
          return mix(mix(hash(i), hash(i+vec2(1,0)), u.x), mix(hash(i+vec2(0,1)), hash(i+vec2(1,1)), u.x), u.y); }
        void main(){
          vec2 p = vW;
          vec2 fw = fwidth(p);
          float ax = abs(p.x);
          bool runway = ax < ${R};
          // ---- base surfaces
          vec3 col;
          if (runway) {
            col = uRunway;
            float seam = aa(abs(fract(p.y / 1.0 + 0.5) - 0.5) - 0.0015, fw.y);   // expansion joints across the runway
            col = mix(col, uInk, seam * 0.05);
            float off = clamp(ax / ${R}, 0.0, 1.0);                               // off-center pain, made visible
            col = mix(col, uInk, off * off * 0.045);
          } else {
            float mottle = (noise(p * 1.7) - 0.5) * 0.035 + (noise(p * 9.0) - 0.5) * 0.015;
            col = uGround * (1.0 + mottle);
            vec2 j = abs(fract(p / 2.0 + 0.5) - 0.5) * 2.0;                       // concrete joints every 2
            float joint = max(aa(j.x - 0.0025, fw.x), aa(j.y - 0.0025, fw.y));
            col = mix(col, uInk, joint * 0.07);
            col = mix(col, uInk, 0.06 * (1.0 - smoothstep(0.0, 0.18, ax - ${R})));   // soft shadow line at the curb foot
          }
          // ---- smell fields
          float cA = 0.0, cB = 0.0;
          for (int i = 0; i < ${N}; i++) {
            if (float(i) >= uN) break;
            vec4 o = uObj[i];
            vec2 d = p - o.xy;
            float c = o.w * exp(-dot(d, d) / (2.0 * ${(this.sigma * this.sigma).toFixed(4)}));
            if (o.z < 1.5) cA += c; else cB += c;
          }
          cA = min(cA, 1.0); cB = min(cB, 1.0);
          vec2 q = mat2(0.7071, -0.7071, 0.7071, 0.7071) * p / 0.034;
          float dl = length(fract(q) - 0.5);
          float dots = aa(dl - 0.44 * sqrt(cA), fwidth(dl)) * step(0.03, cA);
          col = mix(col, uInk, dots * 0.34);
          float h = dot(p, vec2(0.5, 0.866)) / 0.026;
          float hatch = aa(abs(fract(h) - 0.5) - 0.2 * cB, fwidth(h)) * step(0.03, cB);
          col = mix(col, mix(uInk, uHoney, 0.7), hatch * 0.3);
          // ---- rewarded center line: dashes glow ahead of the robot while it earns reward
          if (runway) {
            float dash = step(0.42, fract(p.y / 0.16));
            float line = aa(ax - 0.0022, fw.x) * dash;
            float ahead = p.y - uFly.y;
            float glow = uReward * smoothstep(-0.3, 0.1, ahead) * (1.0 - smoothstep(0.4, 2.2, ahead));
            col = mix(col, mix(uInk, uAccent, glow), line * mix(0.45, 0.95, glow));
          }
          // ---- pain ripple
          float pr = length(p - uPainAt);
          float ring = uPain * aa(abs(pr - (1.0 - uPain) * 0.5 - 0.12) - 0.004, fwidth(pr));
          col = mix(col, uAccent, ring * 0.9);
          // ---- haze into the horizon
          col = mix(col, uHorizon, smoothstep(${FOG_NEAR.toFixed(1)}, ${FOG_FAR.toFixed(1)}, vDepth));
          gl_FragColor = vec4(col, 1.0);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
    });
    this.floor = new THREE.Mesh(new THREE.PlaneGeometry(60, 60), mat);
    this.floor.rotation.x = -Math.PI / 2;
    this.scene.add(this.floor);

    this.shadowFloor = new THREE.Mesh(new THREE.PlaneGeometry(60, 60), new THREE.ShadowMaterial({ opacity: 0.16, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1 }));
    this.shadowFloor.rotation.x = -Math.PI / 2;
    this.shadowFloor.receiveShadow = true;
    this.scene.add(this.shadowFloor);
  }

  buildCurbs() {
    const len = 44;
    const curbMat = new THREE.MeshStandardMaterial({ color: 0xf7f6f2, roughness: 0.55 });
    this.curbs = [];
    for (const s of [1, -1]) {
      const g = new THREE.Group();
      const curb = new THREE.Mesh(new THREE.BoxGeometry(CURB_W, CURB_H, len), curbMat);
      curb.position.set(s * (this.road + CURB_W / 2), CURB_H / 2, 0);
      curb.castShadow = true; curb.receiveShadow = true;
      g.add(curb);
      // light strip on the inner face: flashes where the robot hits it
      const u = { uHitZ: { value: -999 }, uFlash: { value: 0 }, uBase: { value: new THREE.Color('#d6d3cb') }, uAccent: { value: ACCENT.clone() } };
      const strip = new THREE.Mesh(new THREE.BoxGeometry(0.004, 0.007, len), new THREE.ShaderMaterial({
        uniforms: u,
        vertexShader: 'varying float vZ; void main(){ vec4 w = modelMatrix * vec4(position,1.0); vZ = w.z; gl_Position = projectionMatrix * viewMatrix * w; }',
        fragmentShader: `uniform float uHitZ, uFlash; uniform vec3 uBase, uAccent; varying float vZ;
          void main(){ float d = vZ - uHitZ; float g = uFlash * exp(-d * d / 0.35);
            gl_FragColor = vec4(mix(uBase, uAccent, clamp(g, 0.0, 1.0)), 1.0);
            #include <tonemapping_fragment>
            #include <colorspace_fragment>
          }`,
      }));
      strip.position.set(s * (this.road - 0.001), CURB_H * 0.72, 0);
      g.add(strip);
      this.scene.add(g);
      this.curbs.push({ side: s, g, u });
    }
  }

  buildNumerals() {
    this.numCache = new Map();
    this.fontReady = false;
    document.fonts?.load('600 180px "Inter"').then(() => { this.fontReady = true; this.numCache.clear(); }).catch(() => { this.fontReady = true; });
    const geo = new THREE.PlaneGeometry(0.62, 0.31);
    this.numerals = [];
    for (let i = 0; i < NUM_N; i++) {
      const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false, opacity: 0.16, color: INK, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 }));
      m.rotation.set(-Math.PI / 2, 0, Math.PI);          // lies flat, reads from behind the robot
      m.visible = false;
      this.scene.add(m);
      this.numerals.push({ m, n: null });
    }
  }

  numeralTexture(n) {
    const key = `${n}:${this.fontReady}`;
    if (this.numCache.has(key)) return this.numCache.get(key);
    const c = document.createElement('canvas'); c.width = 512; c.height = 256;
    const g = c.getContext('2d');
    g.fillStyle = '#fff';
    g.textAlign = 'center'; g.textBaseline = 'middle';
    g.font = '600 180px Inter, ui-sans-serif, system-ui, sans-serif';
    g.fillText(String(n).padStart(2, '0'), 256, 140);
    const t = new THREE.CanvasTexture(c);
    t.anisotropy = 8;
    if (this.numCache.size > 60) this.numCache.clear();
    this.numCache.set(key, t);
    return t;
  }

  flashCurb(side, z) {
    const c = this.curbs.find((k) => k.side === side);
    if (c) { c.u.uHitZ.value = z; c.u.uFlash.value = 1; }
  }

  // fly: { x, z }, reward 0..1
  update(dt, fly, reward, camera) {
    const snap = (v, s) => Math.round(v / s) * s;
    this.floor.position.set(snap(fly.x, 5), 0, snap(fly.z, 5));
    this.shadowFloor.position.set(snap(fly.x, 5), 0.0005, snap(fly.z, 5));
    this.sky.position.copy(camera.position);
    this.floorU.uFly.value.set(fly.x, fly.z);
    this.floorU.uReward.value += (reward - this.floorU.uReward.value) * (1 - Math.exp(-dt * 4));
    for (const c of this.curbs) {
      c.g.position.z = snap(fly.z, 5) + 12;
      c.u.uFlash.value = Math.max(0, c.u.uFlash.value - dt * 1.4);
    }
    // painted distance numerals on the right half of the runway
    const n0 = Math.floor((fly.z - 2) / NUM_EVERY);
    this.numerals.forEach((N, i) => {
      const k = n0 + i, z = k * NUM_EVERY;
      if (k < 1) { N.m.visible = false; return; }
      if (N.n !== k || N.font !== this.fontReady) {
        N.n = k; N.font = this.fontReady;
        N.m.material.alphaMap = this.numeralTexture(k * NUM_EVERY);
        N.m.material.needsUpdate = true;
      }
      N.m.visible = true;
      N.m.position.set(-(this.road - 0.34), 0.0012, z);
    });
  }
}
