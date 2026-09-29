// Jev as the robot fly's descending motor system: senses + mushroom-body
// output in, one typed decision out. Up to two calls in flight.
// Vision arrives as real LC/LPLC neuron activity (see eye.js), never as object
// labels: what an object *means* only reaches Jev through the learned odor valence.

export const ACTIONS = {
  walk_forward:  { label: 'Walk forward', dn: 'DNp09', circuit: 'walk_forward', desc: 'Walk forward (DNp09)' },
  veer_left:     { label: 'Veer left', dn: 'DNa02 · L', circuit: 'turn_left', desc: 'Veer left (left DNa02)' },
  veer_right:    { label: 'Veer right', dn: 'DNa02 · R', circuit: 'turn_right', desc: 'Veer right (right DNa02)' },
  walk_backward: { label: 'Back away', dn: 'MDN', circuit: 'walk_backward', desc: 'Back away (MDN)' },
  takeoff:       { label: 'Boosters', dn: 'DNp01', circuit: 'takeoff', desc: 'Fire boosters and fly forward (DNp01)' },
  feed:          { label: 'Feed', dn: 'Fdg', circuit: 'feed', desc: 'Stop and feed (Fdg)' },
};

// Short literal rules, first match wins. Everything the rules mention is a field in the state.
const INSTRUCTIONS = [
  'You steer a robot fly walking along a road. Pick its next motor program using the first rule that fits.',
  '1. sugar_under_feet is true and you are hungry: feed.',
  '2. pain is strong: back away.',
  '3. curb is on your LEFT: veer right. curb is on your RIGHT: veer left.',
  '4. A smell that means aversive: if it is stronger on one side, veer to the other side; if it is equal on both sides and vision shows something ahead (any of its middle three numbers is 3 or more), veer toward the side whose vision numbers are lower, and if both sides are equal, veer left.',
  '5. A smell that means attractive is stronger on one side and you are hungry: veer to that side.',
  '6. stuck is yes: pick the command with the highest descending_drive.',
  '7. vision shows something ahead and its smell means unknown: walk forward to find out what it is.',
  '8. Otherwise follow the center line: to your RIGHT: veer right. To your LEFT: veer left. You are on it: walk forward.',
].join(' ');

const QUESTIONS = {
  action: { type: 'choice', instructions: INSTRUCTIONS, criteria: Object.fromEntries(Object.entries(ACTIONS).map(([k, a]) => [k, a.desc])) },
};

const PRICE_PER_TOKEN = 0.042 / 1e6;

const meaning = (v) => (v <= -0.08 ? 'aversive' : v >= 0.08 ? 'attractive' : 'unknown');

export function buildState(s, mb) {
  const smell = {};
  for (const [o, name] of [['A', 'smell_A'], ['B', 'smell_B']]) {
    const { left, right } = s.antennae[o];
    const peak = Math.max(left, right);
    if (peak < 0.05) { smell[name] = 'none'; continue; }
    const diff = left - right;
    smell[name] = {
      stronger_on: Math.abs(diff) < 0.05 * peak + 0.008 ? 'both sides equally' : diff > 0 ? 'LEFT' : 'RIGHT',
      means: meaning(mb.valence(o).value),
    };
  }
  return {
    currently_doing: `${ACTIONS[s.program]?.label.toLowerCase() || s.program} for ${s.programT.toFixed(1)} s`,
    hungry: s.fed < 0.7,
    sugar_under_feet: s.onSugar,
    pain: s.pain ? (s.pain.startsWith('strong') ? 'strong' : 'mild') : 'none',
    curb: s.edgeLeft ? 'on your LEFT' : s.edgeRight ? 'on your RIGHT' : 'not touching',
    ...smell,
    stuck: s.stalled > 1.5 && !s.onSugar ? `yes, no forward progress for ${Math.round(s.stalled)} s` : 'no',
    vision: s.vision ? s.vision.levels.join(' ') : '0 0 0 0 0 0 0 0 0',
    ...(s.visuomotor ? { descending_drive: s.visuomotor } : {}),
    center_line: Math.abs(s.lane.x) < 0.035 ? 'you are on it' : `to your ${s.lane.x > 0 ? 'RIGHT' : 'LEFT'} (off-center pain ${Math.round(s.lane.pain * 100)}%)`,
  };
}

export class JevDriver {
  constructor({ getSense, mb, onDecision, onStatus }) {
    Object.assign(this, { getSense, mb, onDecision, onStatus });
    this.inflight = 0; this.lastSent = 0; this.seq = 0; this.applied = 0;
    this.count = 0; this.tokens = 0; this.latencies = []; this.errors = 0;
    this.running = false; this.gen = 0;
    this.maxInflight = 2; this.minGap = 160;
  }

  start() {
    if (this.running) return;
    this.running = true;
    const gen = ++this.gen;
    const tick = () => {
      if (!this.running || gen !== this.gen) return;
      if (this.inflight < this.maxInflight && performance.now() - this.lastSent > this.minGap) this.send();
      setTimeout(tick, 30);
    };
    tick();
  }

  stop() { this.running = false; }

  async send() {
    const sense = this.getSense();
    if (!sense || sense.airborne) return;          // no frames drawn (hidden tab) → don't spend calls
    const state = buildState(sense, this.mb);
    const id = ++this.seq;
    this.inflight++;
    this.lastSent = performance.now();
    const t0 = performance.now();
    try {
      const res = await fetch('/api/decide', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ state, questions: QUESTIONS }) });
      const data = await res.json();
      if (!res.ok || !data.answers?.action) throw new Error(data.error || data.detail?.[0]?.msg || `HTTP ${res.status}`);
      this.errors = 0;
      const rtt = performance.now() - t0;
      this.latencies.push(data.latency_ms ?? rtt);
      if (this.latencies.length > 60) this.latencies.shift();
      this.count++;
      this.tokens += data.usage?.input_tokens || 0;
      this.onStatus?.({ ok: true });
      if (id < this.applied || !this.running) return;
      this.applied = id;
      const a = data.answers;
      this.onDecision({
        action: policy(a.action.probabilities, a.action.choice, sense.stalled || 0),
        probabilities: a.action.probabilities,
        confidence: a.action.confidence,
        latency: data.latency_ms ?? Math.round(rtt),
        state,
      });
    } catch (err) {
      this.errors++;
      this.onStatus?.({ ok: false, error: String(err.message || err) });
      if (this.errors > 3) await new Promise((r) => setTimeout(r, 1500));
    } finally {
      this.inflight--;
    }
  }

  get cost() { return this.tokens * PRICE_PER_TOKEN; }
  get latency() { return this.latencies.at(-1) ?? null; }
  get avgLatency() { return this.latencies.length ? Math.round(this.latencies.reduce((a, b) => a + b, 0) / this.latencies.length) : null; }
}

// While the robot is making progress it does exactly what Jev picked. When it is
// stuck (no forward progress for a while) it explores: each action is taken with
// the probability Jev gives it, so options Jev rates as unlikely still get tried.
// Behavioural variability rising under failure is what animals do too. Nothing
// here holds or overrides a previous choice.
const EXPLORE_AFTER = 1.5;   // s without forward progress
function policy(probs, choice, stalled) {
  if (!probs || stalled < EXPLORE_AFTER) return choice;
  const entries = Object.entries(probs);
  const total = entries.reduce((sum, [, p]) => sum + Math.max(p, 0), 0);
  if (!total) return choice;
  let r = Math.random() * total;
  for (const [k, p] of entries) { r -= Math.max(p, 0); if (r <= 0) return k; }
  return choice;
}
