/**
 * PCM Injector AudioWorklet — receives float32 PCM samples via port.postMessage
 * and outputs them to the AudioContext graph (→ AnalyserNode).
 *
 * The main thread receives WebSocket binary frames (float32 LE) and forwards
 * them here via the MessagePort. This worklet buffers incoming samples and
 * drains them in 128-sample render quanta.
 */
class PcmInjectorProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.buffer = new Float32Array(0);
        this.port.onmessage = (e) => {
            // e.data is a Float32Array of PCM samples
            const incoming = e.data;
            const merged = new Float32Array(this.buffer.length + incoming.length);
            merged.set(this.buffer);
            merged.set(incoming, this.buffer.length);
            this.buffer = merged;
        };
    }

    process(inputs, outputs) {
        const output = outputs[0];
        if (!output || output.length === 0) return true;

        const channel = output[0];
        const needed = channel.length; // 128 samples per quantum

        if (this.buffer.length >= needed) {
            channel.set(this.buffer.subarray(0, needed));
            this.buffer = this.buffer.subarray(needed);
        } else if (this.buffer.length > 0) {
            // Partial fill — pad remainder with silence
            channel.set(this.buffer);
            this.buffer = new Float32Array(0);
        }
        // else: output stays silent (zeros)

        return true;
    }
}

registerProcessor('pcm-injector-processor', PcmInjectorProcessor);
