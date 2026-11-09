// The squeal: louder, higher and rougher as the log posterior drops below its
// value at theta-hat. Driven by level L = log10(1 + cost), so a cost of 10
// is a faint whine and a cost of 10^4 is a shriek.

export function squealLevel(cost: number): number {
  return Math.log10(1 + Math.max(0, cost));
}

export class Squeal {
  private ctx: AudioContext | null = null;
  private oscA!: OscillatorNode;
  private oscB!: OscillatorNode;
  private lfo!: OscillatorNode;
  private lfoGain!: GainNode;
  private filter!: BiquadFilterNode;
  private amp!: GainNode;
  private trem!: GainNode;
  muted = false;

  // Must be called from a user gesture (browser autoplay policy).
  start() {
    if (this.ctx) {
      if (this.ctx.state === "suspended") void this.ctx.resume();
      return;
    }
    const ctx = new AudioContext();
    this.ctx = ctx;
    this.oscA = new OscillatorNode(ctx, { type: "sawtooth", frequency: 220 });
    this.oscB = new OscillatorNode(ctx, { type: "sawtooth", frequency: 221 });
    this.filter = new BiquadFilterNode(ctx, { type: "lowpass", frequency: 1000, Q: 4 });
    this.trem = new GainNode(ctx, { gain: 1 });
    this.lfo = new OscillatorNode(ctx, { type: "sine", frequency: 6 });
    this.lfoGain = new GainNode(ctx, { gain: 0 });
    this.amp = new GainNode(ctx, { gain: 0 });
    this.lfo.connect(this.lfoGain).connect(this.trem.gain);
    this.oscA.connect(this.filter);
    this.oscB.connect(this.filter);
    this.filter.connect(this.trem).connect(this.amp).connect(ctx.destination);
    this.oscA.start();
    this.oscB.start();
    this.lfo.start();
  }

  // holding: the user is dragging, so sustain; otherwise sound briefly and fade.
  update(cost: number, holding: boolean) {
    const ctx = this.ctx;
    if (!ctx) return;
    const L = squealLevel(cost);
    const now = ctx.currentTime;
    const f = 180 * Math.pow(2, 0.9 * L);
    const tc = 0.02;
    this.oscA.frequency.setTargetAtTime(f, now, tc);
    this.oscB.frequency.setTargetAtTime(f * (1 + 0.012 * L), now, tc);
    this.filter.frequency.setTargetAtTime(600 + 1800 * L, now, tc);
    this.lfo.frequency.setTargetAtTime(5 + 3 * L, now, tc);
    this.lfoGain.gain.setTargetAtTime(Math.min(0.5, 0.1 * L), now, tc);
    const g = this.muted || cost < 0.5 ? 0 : Math.min(0.22, 0.03 + 0.05 * L);
    const amp = this.amp.gain;
    amp.cancelScheduledValues(now);
    amp.setTargetAtTime(g, now, tc);
    if (!holding) amp.setTargetAtTime(0, now + 0.35, 0.12);
  }

  silence() {
    if (!this.ctx) return;
    const now = this.ctx.currentTime;
    this.amp.gain.cancelScheduledValues(now);
    this.amp.gain.setTargetAtTime(0, now, 0.05);
  }
}
