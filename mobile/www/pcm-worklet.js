// Resample the mic to 16 kHz mono, Float32 -> Int16, post PCM chunks (for Vosk).
class PCM16Downsampler extends AudioWorkletProcessor {
  constructor() { super(); this.targetRate = 16000; this._pos = 0; }
  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const ch = input[0];                          // mono: first channel
    const ratio = sampleRate / this.targetRate;   // e.g. 48000/16000 = 3
    const out = [];
    let pos = this._pos;
    while (pos < ch.length) {
      const i = Math.floor(pos), frac = pos - i;
      const s0 = ch[i], s1 = (i + 1 < ch.length) ? ch[i + 1] : s0;
      let s = s0 + (s1 - s0) * frac;               // linear interpolation
      s = Math.max(-1, Math.min(1, s));
      out.push(s < 0 ? s * 0x8000 : s * 0x7fff);   // Float32 -> Int16
      pos += ratio;
    }
    this._pos = pos - ch.length;                   // carry fractional remainder
    if (out.length) {
      const pcm = Int16Array.from(out);
      this.port.postMessage(pcm.buffer, [pcm.buffer]);
    }
    return true;
  }
}
registerProcessor('pcm16-downsampler', PCM16Downsampler);
