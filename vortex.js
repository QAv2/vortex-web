// ╔══════════════════════════════════════════════════════════════════╗
// ║  VORTEX — Real-time Music Visualizer                            ║
// ║  Web port of Python/pygame-ce original                          ║
// ║  WebAudio API + Canvas 2D                                       ║
// ╚══════════════════════════════════════════════════════════════════╝

// ─── Constants ───────────────────────────────────────────────────
const NUM_BANDS = 64;
const FFT_SIZE = 2048;
const FREQ_MIN = 20;
const FREQ_MAX = 16000;
const BEAT_MIN_INTERVAL = 0.2;
const BEAT_THRESHOLD_MULT = 1.5;
const PEAK_FLOOR = 0.001;
const PEAK_HEADROOM = 1.2;
const BG = [8, 8, 12];

// ─── Utilities ───────────────────────────────────────────────────
function rgb(r, g, b) { return `rgb(${r|0},${g|0},${b|0})`; }
function rgba(r, g, b, a) { return `rgba(${r|0},${g|0},${b|0},${a})`; }
function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

// Spectrum gradient: band index 0–63 → [r, g, b]
function spectrumColor(i) {
    const t = i / 63;
    let r, g, b;
    if (t < 0.25) {
        const s = t / 0.25;
        r = 20 * (1 - s); g = 80 + 175 * s; b = 200 + 55 * s;
    } else if (t < 0.5) {
        const s = (t - 0.25) / 0.25;
        r = 20 * s; g = 255 - 55 * s; b = 255 - 155 * s;
    } else if (t < 0.75) {
        const s = (t - 0.5) / 0.25;
        r = 20 + 235 * s; g = 200 + 55 * s; b = 100 - 80 * s;
    } else {
        const s = (t - 0.75) / 0.25;
        r = 255; g = 255 - 155 * s; b = 20 + 180 * s;
    }
    return [Math.round(r), Math.round(g), Math.round(b)];
}

const SPECTRUM_COLORS = Array.from({ length: 64 }, (_, i) => spectrumColor(i));

// Spectrogram heatmap: 256 entries
const SPECTRO_COLORMAP = (() => {
    const map = [];
    for (let i = 0; i < 256; i++) {
        const t = i / 255;
        let r, g, b;
        if (t < 0.15) {
            const s = t / 0.15;
            r = 0; g = 0; b = 180 * s;
        } else if (t < 0.35) {
            const s = (t - 0.15) / 0.20;
            r = 0; g = 255 * s; b = 180 + 75 * s;
        } else if (t < 0.55) {
            const s = (t - 0.35) / 0.20;
            r = 0; g = 255; b = 255 * (1 - s);
        } else if (t < 0.75) {
            const s = (t - 0.55) / 0.20;
            r = 255 * s; g = 255; b = 0;
        } else if (t < 0.90) {
            const s = (t - 0.75) / 0.15;
            r = 255; g = 255; b = 255 * s;
        } else {
            r = 255; g = 255; b = 255;
        }
        map.push([Math.round(r), Math.round(g), Math.round(b)]);
    }
    return map;
})();

function makeTrail(w, h) {
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    const ctx = c.getContext('2d');
    ctx.fillStyle = rgb(...BG);
    ctx.fillRect(0, 0, w, h);
    return { canvas: c, ctx };
}

function formatTime(s) {
    if (!s || isNaN(s)) return '0:00';
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, '0')}`;
}

// ═════════════════════════════════════════════════════════════════
// AUDIO ENGINE
// ═════════════════════════════════════════════════════════════════
class AudioEngine {
    constructor() {
        this.actx = null;
        this.analyser = null;
        this.gainNode = null;
        this.source = null;
        this.audio = null;
        this.tracks = [];
        this.currentTrack = 0;
        this.volume = 0.75;
        this.shuffle = false;
        this.repeat = false;
        this.freqData = null;
        this.timeData = null;
        this.bandEdges = [];
        this.smoothSpectrum = new Float32Array(NUM_BANDS);
        this.prevFreqData = null;
        this.bandPeaks = new Float32Array(NUM_BANDS).fill(0.1);
        this.fluxHistory = [];
        this.lastBeatTime = 0;
        this.beatPulse = 0;
        this.beatDetected = false;
        this.data = {
            spectrum: this.smoothSpectrum,
            waveform: null,
            bass: 0, mid: 0, treble: 0, sub: 0,
            energy: 0, beatDetected: false, beatPulse: 0
        };
        this.onTrackChange = null;

        // Live capture state
        this.mode = 'file';           // 'file' | 'live-system' | 'live-display' | 'live-mic'
        this.liveSource = null;       // MediaStreamAudioSourceNode for live modes
        this.liveStream = null;       // MediaStream reference (for cleanup)
        this.isCapturing = false;
        this.onLiveCaptureEnd = null;
    }

    _computeBandEdges() {
        const edges = [];
        const ratio = FREQ_MAX / FREQ_MIN;
        const binHz = (this.actx?.sampleRate || 44100) / FFT_SIZE;
        for (let i = 0; i <= NUM_BANDS; i++) {
            const freq = FREQ_MIN * Math.pow(ratio, i / NUM_BANDS);
            edges.push(Math.round(freq / binHz));
        }
        return edges;
    }

    async init() {
        if (this.actx) return;
        this.actx = new (window.AudioContext || window.webkitAudioContext)();
        if (this.actx.state === 'suspended') await this.actx.resume();

        this.analyser = this.actx.createAnalyser();
        this.analyser.fftSize = FFT_SIZE;
        this.analyser.smoothingTimeConstant = 0;
        this.analyser.minDecibels = -100;
        this.analyser.maxDecibels = -10;

        this.gainNode = this.actx.createGain();
        this.gainNode.gain.value = this.volume;
        this.analyser.connect(this.gainNode);
        this.gainNode.connect(this.actx.destination);

        this.freqData = new Uint8Array(this.analyser.frequencyBinCount);
        this.timeData = new Float32Array(this.analyser.fftSize);
        this.prevFreqData = new Float32Array(this.analyser.frequencyBinCount);
        this.bandEdges = this._computeBandEdges();

        this.audio = document.createElement('audio');
        this.audio.addEventListener('ended', () => {
            if (this.repeat) { this.audio.currentTime = 0; this.audio.play(); }
            else this.nextTrack();
        });
    }

    loadFiles(files) {
        const start = this.tracks.length;
        const audioExt = /\.(mp3|wav|ogg|flac|m4a|aac|opus|webm|wma|alac|caf|aif|aiff)$/i;
        for (const file of files) {
            if (file.type.startsWith('audio/') || file.type.startsWith('video/') || audioExt.test(file.name) || file.type === '') {
                this.tracks.push({ name: file.name.replace(/\.[^.]+$/, ''), url: URL.createObjectURL(file) });
            }
        }
        if (this.tracks.length > start && (!this.audio.src || this.audio.paused)) {
            this.play(start);
        }
    }

    async play(index) {
        if (index !== undefined) this.currentTrack = index;
        if (this.tracks.length === 0) return;
        const track = this.tracks[this.currentTrack];
        if (this.audio.src !== track.url) this.audio.src = track.url;
        if (!this.source) {
            this.source = this.actx.createMediaElementSource(this.audio);
            this.source.connect(this.analyser);
        }
        // Resume AudioContext if suspended (mobile autoplay policy)
        if (this.actx.state === 'suspended') await this.actx.resume();
        try { await this.audio.play(); } catch (e) { console.warn('Playback blocked:', e); }
        if (this.onTrackChange) this.onTrackChange();
    }

    async toggle() {
        if (!this.audio?.src) return;
        if (this.actx.state === 'suspended') await this.actx.resume();
        this.audio.paused ? await this.audio.play() : this.audio.pause();
    }

    seek(frac) {
        if (this.audio?.duration) this.audio.currentTime = frac * this.audio.duration;
    }

    setVolume(v) {
        this.volume = clamp(v, 0, 1);
        if (this.gainNode) this.gainNode.gain.value = this.volume;
    }

    nextTrack() {
        if (this.tracks.length === 0) return;
        this.currentTrack = this.shuffle
            ? Math.floor(Math.random() * this.tracks.length)
            : (this.currentTrack + 1) % this.tracks.length;
        this.play();
    }

    prevTrack() {
        if (this.tracks.length === 0) return;
        if (this.audio.currentTime > 3) { this.audio.currentTime = 0; return; }
        this.currentTrack = (this.currentTrack - 1 + this.tracks.length) % this.tracks.length;
        this.play();
    }

    get playing() { return this.isCapturing || (this.audio && !this.audio.paused); }
    get currentTime() { return this.audio?.currentTime || 0; }
    get duration() { return this.audio?.duration || 0; }
    get trackName() { return this.tracks[this.currentTrack]?.name || ''; }
    get trackCount() { return this.tracks.length; }

    update(dt) {
        if (!this.analyser || !this.playing) {
            const decay = Math.pow(0.85, dt * 30);
            this.beatPulse *= decay;
            for (let i = 0; i < NUM_BANDS; i++) this.smoothSpectrum[i] *= decay;
            this.data.bass *= decay;
            this.data.mid *= decay;
            this.data.treble *= decay;
            this.data.sub *= decay;
            this.data.energy *= decay;
            this.data.beatDetected = false;
            this.data.beatPulse = this.beatPulse;
            return this.data;
        }

        this.analyser.getByteFrequencyData(this.freqData);
        this.analyser.getFloatTimeDomainData(this.timeData);

        // Map to 64 log-spaced bands
        const raw = new Float32Array(NUM_BANDS);
        for (let i = 0; i < NUM_BANDS; i++) {
            const lo = this.bandEdges[i];
            const hi = Math.max(lo + 1, this.bandEdges[i + 1]);
            let sum = 0;
            for (let j = lo; j < hi && j < this.freqData.length; j++) sum += this.freqData[j];
            raw[i] = sum / ((hi - lo) * 255);
        }

        // Per-band peak normalization
        const peakDecay = Math.pow(0.999, dt * 30);
        const attack = 1 - Math.pow(1 - 0.6, dt * 30);
        const decay = Math.pow(0.85, dt * 30);
        for (let i = 0; i < NUM_BANDS; i++) {
            if (raw[i] > this.bandPeaks[i]) this.bandPeaks[i] = raw[i];
            else {
                this.bandPeaks[i] *= peakDecay;
                if (this.bandPeaks[i] < PEAK_FLOOR) this.bandPeaks[i] = PEAK_FLOOR;
            }
            const norm = clamp(raw[i] / (this.bandPeaks[i] * PEAK_HEADROOM), 0, 1);
            if (norm > this.smoothSpectrum[i])
                this.smoothSpectrum[i] += (norm - this.smoothSpectrum[i]) * attack;
            else
                this.smoothSpectrum[i] *= decay;
        }

        // Beat detection (spectral flux)
        let flux = 0;
        for (let i = 0; i < this.freqData.length; i++) {
            const diff = this.freqData[i] / 255 - this.prevFreqData[i];
            if (diff > 0) flux += diff;
        }
        for (let i = 0; i < this.freqData.length; i++) this.prevFreqData[i] = this.freqData[i] / 255;

        this.fluxHistory.push(flux);
        if (this.fluxHistory.length > 60) this.fluxHistory.shift();

        const now = performance.now() / 1000;
        this.beatDetected = false;
        if (this.fluxHistory.length > 8) {
            let mean = 0;
            for (const f of this.fluxHistory) mean += f;
            mean /= this.fluxHistory.length;
            let variance = 0;
            for (const f of this.fluxHistory) variance += (f - mean) ** 2;
            const std = Math.sqrt(variance / this.fluxHistory.length);
            if (flux > mean + BEAT_THRESHOLD_MULT * std && (now - this.lastBeatTime) > BEAT_MIN_INTERVAL) {
                this.beatDetected = true;
                this.beatPulse = 1.0;
                this.lastBeatTime = now;
            }
        }

        this.beatPulse *= Math.pow(0.85, dt * 30);
        if (this.beatPulse < 0.01) this.beatPulse = 0;

        // Band energies
        const avg = (lo, hi) => {
            let s = 0, c = hi - lo;
            for (let i = lo; i < hi && i < NUM_BANDS; i++) s += this.smoothSpectrum[i];
            return c > 0 ? s / c : 0;
        };

        this.data.spectrum = this.smoothSpectrum;
        this.data.waveform = this.timeData;
        this.data.sub = avg(0, 5);
        this.data.bass = avg(0, 15);
        this.data.mid = avg(15, 45);
        this.data.treble = avg(45, 64);
        this.data.energy = avg(0, 64);
        this.data.beatDetected = this.beatDetected;
        this.data.beatPulse = this.beatPulse;
        return this.data;
    }

    // ── Live Capture ─────────────────────────────────────────────

    async startLiveCapture(stream, modeName) {
        if (!this.actx) await this.init();

        // Disconnect file source from analyser (don't destroy it)
        if (this.source) {
            try { this.source.disconnect(this.analyser); } catch (e) {}
        }

        this.liveStream = stream;
        this.liveSource = this.actx.createMediaStreamSource(stream);
        this.liveSource.connect(this.analyser);

        // Mute output — the source app is already playing audio
        this._savedVolume = this.gainNode.gain.value;
        this.gainNode.gain.value = 0;

        this.mode = modeName;
        this.isCapturing = true;
    }

    stopLiveCapture() {
        if (this.liveSource) {
            try { this.liveSource.disconnect(this.analyser); } catch (e) {}
            this.liveSource = null;
        }
        if (this.liveStream) {
            this.liveStream.getTracks().forEach(t => t.stop());
            this.liveStream = null;
        }

        // Clean up system capture (WebSocket + worklet + native service)
        if (this._captureWs) {
            try { this._captureWs.close(); } catch (e) {}
            this._captureWs = null;
        }
        if (this.pcmWorkletNode) {
            try { this.pcmWorkletNode.disconnect(this.analyser); } catch (e) {}
            this.pcmWorkletNode = null;
        }
        if (this.mode === 'live-system' && typeof Capacitor !== 'undefined' && Capacitor.Plugins?.AudioCapture) {
            Capacitor.Plugins.AudioCapture.stopCapture().catch(() => {});
        }

        this.isCapturing = false;
        this.mode = 'file';

        // Restore volume and reconnect file source
        this.gainNode.gain.value = this._savedVolume ?? this.volume;
        if (this.source) {
            try { this.source.connect(this.analyser); } catch (e) {}
        }
    }

    async startDisplayCapture() {
        // Firefox does not support audio capture via getDisplayMedia
        const isFirefox = navigator.userAgent.includes('Firefox');
        if (isFirefox) {
            throw new Error('Screen audio capture is not supported in Firefox. Use Chrome or Edge, or try Microphone mode.');
        }

        const stream = await navigator.mediaDevices.getDisplayMedia({
            video: true,
            audio: true,
            preferCurrentTab: false,
            systemAudio: 'include'
        });

        // Drop video track — we only want audio
        stream.getVideoTracks().forEach(t => t.stop());

        const audioTracks = stream.getAudioTracks();
        if (audioTracks.length === 0) {
            stream.getTracks().forEach(t => t.stop());
            throw new Error('No audio track. Make sure to check "Share audio" in the share dialog.');
        }

        const audioStream = new MediaStream(audioTracks);

        // Handle user clicking browser's "Stop sharing" button
        audioTracks[0].addEventListener('ended', () => {
            this.stopLiveCapture();
            if (this.onLiveCaptureEnd) this.onLiveCaptureEnd();
        });

        await this.startLiveCapture(audioStream, 'live-display');
    }

    async startMicCapture() {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false
            }
        });

        stream.getAudioTracks()[0].addEventListener('ended', () => {
            this.stopLiveCapture();
            if (this.onLiveCaptureEnd) this.onLiveCaptureEnd();
        });

        // For mic mode, don't mute — there's no other app playing through speakers
        await this.startLiveCapture(stream, 'live-mic');
        this.gainNode.gain.value = 0; // still mute to prevent feedback loop
    }

    async startSystemCapture() {
        // Native Android AudioPlaybackCapture via Capacitor plugin
        if (typeof Capacitor === 'undefined' || !Capacitor.Plugins?.AudioCapture) {
            throw new Error('System audio capture is only available in the Android app');
        }

        const result = await Capacitor.Plugins.AudioCapture.startCapture();
        const wsPort = result.port || 8765;

        if (!this.actx) await this.init();

        // Load the PCM injector worklet
        await this.actx.audioWorklet.addModule('pcm-injector-worklet.js');
        this.pcmWorkletNode = new AudioWorkletNode(this.actx, 'pcm-injector-processor');

        // Disconnect file source from analyser
        if (this.source) {
            try { this.source.disconnect(this.analyser); } catch (e) {}
        }

        // Connect worklet → analyser
        this.pcmWorkletNode.connect(this.analyser);

        // Mute output — the source app is already playing audio
        this._savedVolume = this.gainNode.gain.value;
        this.gainNode.gain.value = 0;

        // Open WebSocket to the native service
        this._captureWs = new WebSocket(`ws://127.0.0.1:${wsPort}`);
        this._captureWs.binaryType = 'arraybuffer';
        this._captureWs.onmessage = (e) => {
            const pcm = new Float32Array(e.data);
            this.pcmWorkletNode.port.postMessage(pcm);
        };
        this._captureWs.onerror = () => {
            this.stopLiveCapture();
            if (this.onLiveCaptureEnd) this.onLiveCaptureEnd();
        };
        this._captureWs.onclose = () => {
            if (this.isCapturing && this.mode === 'live-system') {
                this.stopLiveCapture();
                if (this.onLiveCaptureEnd) this.onLiveCaptureEnd();
            }
        };

        this.mode = 'live-system';
        this.isCapturing = true;
    }

    static detectCapabilities() {
        const isCapacitor = typeof Capacitor !== 'undefined' && Capacitor.isNativePlatform?.();
        const isAndroid = isCapacitor && Capacitor.getPlatform() === 'android';
        return {
            filePlayback: true,
            systemCapture: isAndroid,
            displayCapture: !isCapacitor && !!(navigator.mediaDevices?.getDisplayMedia),
            micCapture: !!(navigator.mediaDevices?.getUserMedia)
        };
    }
}


// ═════════════════════════════════════════════════════════════════
// PRESET 1: SPECTRUM BARS
// ═════════════════════════════════════════════════════════════════
function createSpectrumBars() {
    const peaks = new Float32Array(64);
    return {
        name: 'Spectrum Bars',
        render(ctx, audio, dt, w, h) {
            ctx.fillStyle = rgb(...BG);
            ctx.fillRect(0, 0, w, h);

            const margin = 40;
            const gap = 2;
            const barW = Math.max(2, ((w - margin * 2) - gap * 63) / 64);
            const yBase = h - margin;
            const areaH = h - margin * 2;

            for (let i = 0; i < 64; i++) {
                const val = audio.spectrum[i];
                const barH = val * 0.8 * areaH;
                const x = margin + i * (barW + gap);
                const [r, g, b] = SPECTRUM_COLORS[i];

                // Main bar
                ctx.fillStyle = rgb(r, g, b);
                ctx.fillRect(x, yBase - barH, barW, barH);

                // Bright top edge
                const tr = Math.min(255, r + 60), tg = Math.min(255, g + 60), tb = Math.min(255, b + 60);
                if (barH > 1) {
                    ctx.fillStyle = rgb(tr, tg, tb);
                    ctx.fillRect(x, yBase - barH, barW, 2);
                }

                // Reflection
                ctx.fillStyle = rgba(r >> 2, g >> 2, b >> 2, 0.5);
                ctx.fillRect(x, yBase + 1, barW, barH / 3);

                // Peak hold
                if (val > peaks[i]) peaks[i] = val;
                else {
                    peaks[i] *= Math.pow(0.97, dt * 30);
                    peaks[i] -= 0.005 * dt * 30;
                    if (peaks[i] < 0) peaks[i] = 0;
                }
                const peakY = yBase - peaks[i] * 0.8 * areaH;
                ctx.fillStyle = rgb(tr, tg, tb);
                ctx.fillRect(x, peakY - 2, barW, 2);
            }

        }
    };
}


// ═════════════════════════════════════════════════════════════════
// PRESET 2: OSCILLOSCOPE
// ═════════════════════════════════════════════════════════════════
function createOscilloscope() {
    let trail, tc;
    return {
        name: 'Oscilloscope',
        render(ctx, audio, dt, w, h) {
            if (!trail || trail.width !== w || trail.height !== h) {
                ({ canvas: trail, ctx: tc } = makeTrail(w, h));
            }
            // Phosphor decay
            const da = 1 - Math.pow(1 - 25 / 255, dt * 30);
            tc.fillStyle = rgba(...BG, da);
            tc.fillRect(0, 0, w, h);

            if (audio.waveform) {
                const margin = 40, cy = h / 2, amp = (h - 80) / 2;
                const wf = audio.waveform;
                const step = Math.max(1, (wf.length / (w - margin * 2)) | 0);
                const beat = audio.beatPulse;
                let r = 40 + (beat > 0.1 ? 200 * beat : 0) | 0;
                let g = 180 + (beat > 0.1 ? 75 * beat : 0) | 0;
                let b = 80 + (beat > 0.1 ? 175 * beat : 0) | 0;
                r = Math.min(255, r); g = Math.min(255, g); b = Math.min(255, b);

                tc.strokeStyle = rgb(r, g, b);
                tc.lineWidth = 2;
                tc.beginPath();
                for (let x = 0; x < w - margin * 2; x++) {
                    const y = cy - wf[Math.min(x * step, wf.length - 1)] * amp;
                    x === 0 ? tc.moveTo(margin, y) : tc.lineTo(margin + x, y);
                }
                tc.stroke();

                // Bright highlight
                tc.strokeStyle = rgb(Math.min(255, r + 80), Math.min(255, g + 80), Math.min(255, b + 80));
                tc.lineWidth = 0.8;
                tc.beginPath();
                for (let x = 0; x < w - margin * 2; x++) {
                    const y = cy - wf[Math.min(x * step, wf.length - 1)] * amp;
                    x === 0 ? tc.moveTo(margin, y) : tc.lineTo(margin + x, y);
                }
                tc.stroke();
            }
            ctx.drawImage(trail, 0, 0);
        }
    };
}


// ═════════════════════════════════════════════════════════════════
// PRESET 3: PARTICLE FOUNTAIN
// ═════════════════════════════════════════════════════════════════
function createParticleFountain() {
    const MAX = 800;
    const px = new Float32Array(MAX), py = new Float32Array(MAX);
    const vx = new Float32Array(MAX), vy = new Float32Array(MAX);
    const life = new Float32Array(MAX), maxLife = new Float32Array(MAX);
    const pr = new Uint8Array(MAX), pg = new Uint8Array(MAX), pb = new Uint8Array(MAX);
    const sz = new Float32Array(MAX);
    let count = 0;
    let spawnAccum = 0;

    function spawn(x, y, velX, velY, lt, r, g, b, s) {
        if (count >= MAX) return;
        const i = count++;
        px[i] = x; py[i] = y; vx[i] = velX; vy[i] = velY;
        life[i] = lt; maxLife[i] = lt;
        pr[i] = r; pg[i] = g; pb[i] = b; sz[i] = s;
    }

    // Pick a color from the spectrum based on which frequencies are loudest
    function freqColor(audio) {
        // Weighted pick across 64 bands
        let sumW = 0;
        for (let i = 0; i < 64; i++) sumW += audio.spectrum[i];
        if (sumW < 0.01) return SPECTRUM_COLORS[0];
        let pick = Math.random() * sumW, acc = 0;
        for (let i = 0; i < 64; i++) {
            acc += audio.spectrum[i];
            if (acc >= pick) return SPECTRUM_COLORS[i];
        }
        return SPECTRUM_COLORS[63];
    }

    return {
        name: 'Particle Fountain',
        render(ctx, audio, dt, w, h) {
            ctx.fillStyle = rgb(...BG);
            ctx.fillRect(0, 0, w, h);

            const cx = w / 2;
            const nozzleY = h * 0.82;
            const energy = audio.energy;
            const bass = audio.bass;
            const beat = audio.beatPulse;

            // ── Main jet: upward spray from center nozzle ──
            // Spawn rate scales with overall energy
            const baseRate = energy * 8 + bass * 4;
            spawnAccum += baseRate * dt * 30;
            const toSpawn = spawnAccum | 0;
            spawnAccum -= toSpawn;

            for (let s = 0; s < toSpawn; s++) {
                const [cr, cg, cb] = freqColor(audio);
                // Cone spread: wider with more energy, ±45° max
                const spread = 0.2 + energy * 0.5 + beat * 0.2;
                const angle = -Math.PI / 2 + (Math.random() - 0.5) * Math.PI * spread;
                const power = 200 + energy * 350 + bass * 150 + beat * 100;
                const speed = power * (0.7 + Math.random() * 0.6);
                const lt = 1.2 + energy * 1.8 + Math.random() * 0.5;
                const size = 2.5 + energy * 3 + Math.random() * 1.5;
                spawn(cx + (Math.random() - 0.5) * 6, nozzleY,
                    Math.cos(angle) * speed, Math.sin(angle) * speed,
                    lt, cr, cg, cb, size);
            }

            // ── Side jets: two angled sprays on strong bass ──
            if (bass > 0.3) {
                const sideRate = (bass - 0.3) * 4;
                const sideCount = (sideRate * dt * 30) | 0;
                for (let s = 0; s < sideCount; s++) {
                    const [cr, cg, cb] = freqColor(audio);
                    for (const dir of [-1, 1]) {
                        const angle = -Math.PI / 2 + dir * (0.6 + Math.random() * 0.3);
                        const speed = 150 + bass * 200 + Math.random() * 80;
                        spawn(cx + dir * 8, nozzleY,
                            Math.cos(angle) * speed, Math.sin(angle) * speed,
                            0.8 + bass * 1.2 + Math.random() * 0.3, cr, cg, cb, 2 + bass * 2);
                    }
                }
            }

            // ── Beat eruption: wide radial burst ──
            if (audio.beatDetected) {
                for (let i = 0; i < 50; i++) {
                    const [cr, cg, cb] = freqColor(audio);
                    const angle = -Math.PI * Math.random(); // upper hemisphere
                    const speed = 250 + Math.random() * 300;
                    spawn(cx + (Math.random() - 0.5) * 10, nozzleY,
                        Math.cos(angle) * speed, Math.sin(angle) * speed,
                        0.8 + Math.random() * 0.8,
                        Math.min(255, cr + 60), Math.min(255, cg + 60), Math.min(255, cb + 60),
                        4 + Math.random() * 3);
                }
            }

            // ── Update & draw particles ──
            const gravity = 280;
            let alive = 0;
            for (let i = 0; i < count; i++) {
                life[i] -= dt;
                if (life[i] <= 0) continue;
                vy[i] += gravity * dt;
                px[i] += vx[i] * dt;
                py[i] += vy[i] * dt;
                // Kill particles that fall below nozzle + some margin
                if (py[i] > nozzleY + 40) { continue; }

                const frac = life[i] / maxLife[i];
                let alpha = 1, s = sz[i];
                if (frac < 0.25) { alpha = frac / 0.25; s *= (0.5 + 0.5 * alpha); }
                // Outer glow
                ctx.fillStyle = rgba(pr[i], pg[i], pb[i], alpha * 0.7);
                ctx.fillRect(px[i] - s / 2, py[i] - s / 2, s, s);
                // Bright core
                if (s > 2) {
                    ctx.fillStyle = rgba(
                        Math.min(255, pr[i] + 100),
                        Math.min(255, pg[i] + 100),
                        Math.min(255, pb[i] + 100),
                        alpha * 0.5);
                    const cs = s * 0.35;
                    ctx.fillRect(px[i] - cs / 2, py[i] - cs / 2, cs, cs);
                }
                // Compact
                px[alive] = px[i]; py[alive] = py[i]; vx[alive] = vx[i]; vy[alive] = vy[i];
                life[alive] = life[i]; maxLife[alive] = maxLife[i];
                pr[alive] = pr[i]; pg[alive] = pg[i]; pb[alive] = pb[i]; sz[alive] = sz[i];
                alive++;
            }
            count = alive;

            // ── Nozzle glow: pulsing light at spawn point ──
            const glowR = 15 + energy * 40 + beat * 20;
            const grad = ctx.createRadialGradient(cx, nozzleY, 0, cx, nozzleY, glowR);
            grad.addColorStop(0, rgba(80, 200, 255, 0.3 + energy * 0.3));
            grad.addColorStop(0.5, rgba(80, 200, 255, 0.1 + energy * 0.1));
            grad.addColorStop(1, rgba(80, 200, 255, 0));
            ctx.fillStyle = grad;
            ctx.fillRect(cx - glowR, nozzleY - glowR, glowR * 2, glowR * 2);

            // ── Pool: subtle mist at base where water lands ──
            const poolW = 80 + energy * 200 + beat * 60;
            const poolGrad = ctx.createRadialGradient(cx, nozzleY + 10, 0, cx, nozzleY + 10, poolW);
            poolGrad.addColorStop(0, rgba(40, 150, 200, energy * 0.12));
            poolGrad.addColorStop(1, rgba(40, 150, 200, 0));
            ctx.fillStyle = poolGrad;
            ctx.fillRect(cx - poolW, nozzleY - 10, poolW * 2, 30);
        }
    };
}


// ═════════════════════════════════════════════════════════════════
// PRESET 4: CIRCULAR WAVEFORM
// ═════════════════════════════════════════════════════════════════
function createCircularWaveform() {
    let trail, tc;
    const NP = 256;
    const cosT = new Float32Array(NP), sinT = new Float32Array(NP);
    for (let i = 0; i < NP; i++) {
        const a = (i / NP) * Math.PI * 2;
        cosT[i] = Math.cos(a); sinT[i] = Math.sin(a);
    }
    return {
        name: 'Circular Waveform',
        render(ctx, audio, dt, w, h) {
            if (!trail || trail.width !== w || trail.height !== h) {
                ({ canvas: trail, ctx: tc } = makeTrail(w, h));
            }
            const da = 1 - Math.pow(1 - 22 / 255, dt * 30);
            tc.fillStyle = rgba(...BG, da);
            tc.fillRect(0, 0, w, h);

            const cx = w / 2, cy = h / 2;
            const baseR = Math.min(w, h) * 0.25;
            const breathing = 1 + audio.bass * 0.25 + audio.beatPulse * 0.15;
            const r = baseR * breathing;

            if (audio.waveform) {
                const wf = audio.waveform;
                const step = Math.max(1, (wf.length / NP) | 0);
                const beat = audio.beatPulse, energy = audio.energy;
                const cr = Math.min(255, 40 + beat * 180 + energy * 30);
                const cg = Math.min(255, 180 + energy * 50);
                const cb = Math.min(255, 80 + energy * 120 + beat * 50);

                tc.strokeStyle = rgb(cr, cg, cb);
                tc.lineWidth = 2;
                tc.beginPath();
                for (let i = 0; i < NP; i++) {
                    const sample = wf[Math.min(i * step, wf.length - 1)];
                    const rad = r + sample * baseR * 0.5;
                    const x = cx + cosT[i] * rad, y = cy + sinT[i] * rad;
                    i === 0 ? tc.moveTo(x, y) : tc.lineTo(x, y);
                }
                tc.closePath();
                tc.stroke();

                if (beat > 0.1) {
                    tc.strokeStyle = rgba(80, 200, 255, beat * 0.14);
                    tc.lineWidth = 4;
                    tc.beginPath();
                    tc.arc(cx, cy, r * 0.5, 0, Math.PI * 2);
                    tc.stroke();
                }
            }
            ctx.drawImage(trail, 0, 0);
        }
    };
}


// ═════════════════════════════════════════════════════════════════
// PRESET 5: RADIAL SPECTRUM
// ═════════════════════════════════════════════════════════════════
function createRadialSpectrum() {
    const peaks = new Float32Array(64);
    return {
        name: 'Radial Spectrum',
        render(ctx, audio, dt, w, h) {
            ctx.fillStyle = rgb(...BG);
            ctx.fillRect(0, 0, w, h);
            const cx = w / 2, cy = h / 2;
            const innerR = 35, maxLen = Math.min(w, h) * 0.38;

            for (let i = 0; i < 64; i++) {
                const val = audio.spectrum[i];
                const len = val * maxLen;
                const angle = (i / 64) * Math.PI;
                const [cr, cg, cb] = SPECTRUM_COLORS[i];
                const cosA = Math.cos(angle), sinA = Math.sin(angle);

                for (const mirror of [1, -1]) {
                    ctx.strokeStyle = rgb(cr, cg, cb);
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.moveTo(cx + cosA * innerR, cy + sinA * innerR * mirror);
                    ctx.lineTo(cx + cosA * (innerR + len), cy + sinA * (innerR + len) * mirror);
                    ctx.stroke();

                    if (len > 5) {
                        ctx.strokeStyle = rgb(Math.min(255, cr + 50), Math.min(255, cg + 50), Math.min(255, cb + 50));
                        ctx.beginPath();
                        ctx.moveTo(cx + cosA * (innerR + len - 3), cy + sinA * (innerR + len - 3) * mirror);
                        ctx.lineTo(cx + cosA * (innerR + len), cy + sinA * (innerR + len) * mirror);
                        ctx.stroke();
                    }
                }

                if (val > peaks[i]) peaks[i] = val;
                else {
                    peaks[i] *= Math.pow(0.97, dt * 30);
                    peaks[i] -= 0.003 * dt * 30;
                    if (peaks[i] < 0) peaks[i] = 0;
                }
                const pd = innerR + peaks[i] * maxLen;
                const tr = Math.min(255, cr + 50), tg = Math.min(255, cg + 50), tb = Math.min(255, cb + 50);
                for (const mirror of [1, -1]) {
                    ctx.fillStyle = rgb(tr, tg, tb);
                    ctx.fillRect(cx + cosA * pd - 1, cy + sinA * pd * mirror - 1, 3, 3);
                }
            }

            // Center ring
            ctx.strokeStyle = rgba(80, 80, 100, 0.3);
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(cx, cy, innerR, 0, Math.PI * 2);
            ctx.stroke();
        }
    };
}


// ═════════════════════════════════════════════════════════════════
// PRESET 6: SPECTROGRAM
// ═════════════════════════════════════════════════════════════════
function createSpectrogram() {
    let buf, bc;
    let scrollAccum = 0;
    return {
        name: 'Spectrogram',
        render(ctx, audio, dt, w, h) {
            if (!buf || buf.width !== w || buf.height !== h) {
                buf = document.createElement('canvas');
                buf.width = w; buf.height = h;
                bc = buf.getContext('2d');
                bc.fillStyle = '#000';
                bc.fillRect(0, 0, w, h);
                scrollAccum = 0;
            }
            scrollAccum += dt * 60;
            const scroll = scrollAccum | 0;
            scrollAccum -= scroll;

            if (scroll > 0) {
                const imgData = bc.getImageData(scroll, 0, w - scroll, h);
                bc.putImageData(imgData, 0, 0);
                bc.fillStyle = '#000';
                bc.fillRect(w - scroll, 0, scroll, h);

                const rowH = Math.ceil(h / 64);
                for (let col = 0; col < scroll; col++) {
                    const x = w - scroll + col;
                    for (let i = 0; i < 64; i++) {
                        const idx = clamp((audio.spectrum[i] * 255) | 0, 0, 255);
                        const [r, g, b] = SPECTRO_COLORMAP[idx];
                        bc.fillStyle = rgb(r, g, b);
                        bc.fillRect(x, h - (i + 1) * rowH, 1, rowH);
                    }
                }
            }
            ctx.drawImage(buf, 0, 0);
        }
    };
}


// ═════════════════════════════════════════════════════════════════
// PRESET 7: LISSAJOUS
// ═════════════════════════════════════════════════════════════════
function createLissajous() {
    let trail, tc;
    return {
        name: 'Lissajous',
        render(ctx, audio, dt, w, h) {
            if (!trail || trail.width !== w || trail.height !== h) {
                ({ canvas: trail, ctx: tc } = makeTrail(w, h));
            }
            const da = 1 - Math.pow(1 - 20 / 255, dt * 30);
            tc.fillStyle = rgba(...BG, da);
            tc.fillRect(0, 0, w, h);

            if (audio.waveform) {
                const wf = audio.waveform, n = wf.length;
                const offset = (n / 4 + audio.mid * n * 0.1) | 0;
                const scale = Math.min(w, h) * 0.35;
                const cx = w / 2, cy = h / 2;
                const numPts = 400, step = Math.max(1, (n / numPts) | 0);
                const beat = audio.beatPulse, energy = audio.energy;
                const cr = Math.min(255, 60 + beat * 180);
                const cg = Math.min(255, 100 + energy * 80);
                const cb = Math.min(255, 180 + energy * 50 + beat * 30);

                tc.strokeStyle = rgb(cr, cg, cb);
                tc.lineWidth = 1.5;
                tc.beginPath();
                for (let p = 0; p < numPts; p++) {
                    const idx = p * step;
                    const x = cx + wf[idx % n] * scale;
                    const y = cy + wf[(idx + offset) % n] * scale;
                    p === 0 ? tc.moveTo(x, y) : tc.lineTo(x, y);
                }
                tc.stroke();

                tc.strokeStyle = rgb(Math.min(255, cr + 60), Math.min(255, cg + 60), Math.min(255, cb + 60));
                tc.lineWidth = 0.5;
                tc.beginPath();
                for (let p = 0; p < numPts; p++) {
                    const idx = p * step;
                    const x = cx + wf[idx % n] * scale;
                    const y = cy + wf[(idx + offset) % n] * scale;
                    p === 0 ? tc.moveTo(x, y) : tc.lineTo(x, y);
                }
                tc.stroke();
            }

            ctx.drawImage(trail, 0, 0);
            // Crosshair
            ctx.strokeStyle = rgba(60, 60, 80, 0.3);
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(w / 2, 0); ctx.lineTo(w / 2, h);
            ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2);
            ctx.stroke();
        }
    };
}


// ═════════════════════════════════════════════════════════════════
// PRESET 8: SACRED GEOMETRY
// ═════════════════════════════════════════════════════════════════
function createSacredGeometry() {
    let trail, tc;
    let rotation = 0;
    const RINGS = [
        { sides: 3, bands: [0, 5], color: [40, 100, 220] },
        { sides: 4, bands: [5, 11], color: [40, 200, 220] },
        { sides: 5, bands: [11, 19], color: [40, 200, 100] },
        { sides: 6, bands: [19, 31], color: [220, 220, 40] },
        { sides: 7, bands: [31, 46], color: [220, 160, 40] },
        { sides: 8, bands: [46, 64], color: [220, 80, 180] },
    ];

    return {
        name: 'Sacred Geometry',
        render(ctx, audio, dt, w, h) {
            if (!trail || trail.width !== w || trail.height !== h) {
                ({ canvas: trail, ctx: tc } = makeTrail(w, h));
            }
            const da = 1 - Math.pow(1 - 15 / 255, dt * 30);
            tc.fillStyle = rgba(...BG, da);
            tc.fillRect(0, 0, w, h);

            const cx = w / 2, cy = h / 2;
            const baseR = 35, ringGap = 38;
            const breathing = 0.85 + audio.energy * 0.4 + audio.beatPulse * 0.1;
            rotation += (0.3 + audio.energy * 0.8) * dt;

            const allVerts = [];
            for (let ri = 0; ri < RINGS.length; ri++) {
                const ring = RINGS[ri];
                const radius = (baseR + ri * ringGap) * breathing;
                const rot = rotation + ri * 0.3;
                let ringEnergy = 0;
                const bc = ring.bands[1] - ring.bands[0];
                for (let b = ring.bands[0]; b < ring.bands[1] && b < 64; b++) ringEnergy += audio.spectrum[b];
                ringEnergy /= bc;

                const brightness = clamp(0.3 + ringEnergy * 2, 0, 1);
                const [cr, cg, cb] = ring.color;
                const fr = cr * brightness, fg = cg * brightness, fb = cb * brightness;

                const verts = [];
                tc.strokeStyle = rgb(fr, fg, fb);
                tc.lineWidth = 1.5;
                tc.beginPath();
                for (let v = 0; v <= ring.sides; v++) {
                    const angle = rot + (v / ring.sides) * Math.PI * 2;
                    const x = cx + Math.cos(angle) * radius;
                    const y = cy + Math.sin(angle) * radius;
                    v === 0 ? tc.moveTo(x, y) : tc.lineTo(x, y);
                    if (v < ring.sides) verts.push([x, y]);
                }
                tc.stroke();
                allVerts.push(verts);

                // Inter-ring connections
                if (ri > 0) {
                    const inner = allVerts[ri - 1];
                    tc.strokeStyle = rgba(fr, fg, fb, 0.2);
                    tc.lineWidth = 0.5;
                    for (const [vx, vy] of verts) {
                        let minD = Infinity, best = inner[0];
                        for (const p of inner) {
                            const d = (vx - p[0]) ** 2 + (vy - p[1]) ** 2;
                            if (d < minD) { minD = d; best = p; }
                        }
                        tc.beginPath();
                        tc.moveTo(vx, vy);
                        tc.lineTo(best[0], best[1]);
                        tc.stroke();
                    }
                }
            }
            ctx.drawImage(trail, 0, 0);
        }
    };
}


// ═════════════════════════════════════════════════════════════════
// PRESET 9: STARFIELD
// ═════════════════════════════════════════════════════════════════
function createStarfield() {
    const N = 300;
    const sx = new Float32Array(N), sy = new Float32Array(N), sz = new Float32Array(N);
    const psx = new Float32Array(N), psy = new Float32Array(N), psz = new Float32Array(N);
    for (let i = 0; i < N; i++) {
        sx[i] = (Math.random() - 0.5) * 2;
        sy[i] = (Math.random() - 0.5) * 2;
        sz[i] = Math.random() * 0.9 + 0.1;
        psx[i] = sx[i]; psy[i] = sy[i]; psz[i] = sz[i];
    }
    let trail, tc;

    return {
        name: 'Starfield',
        render(ctx, audio, dt, w, h) {
            if (!trail || trail.width !== w || trail.height !== h) {
                ({ canvas: trail, ctx: tc } = makeTrail(w, h));
            }
            const energy = audio.energy, beat = audio.beatPulse;
            const dimVal = 30 + energy * 25;
            const da = 1 - Math.pow(1 - dimVal / 255, dt * 30);
            tc.fillStyle = rgba(...BG, da);
            tc.fillRect(0, 0, w, h);

            const cx = w / 2, cy = h / 2;
            const fov = Math.min(w, h) * 0.5;
            const speed = (0.25 + energy * 0.6 + (audio.beatDetected ? 0.4 : 0)) * dt;
            const br = Math.min(255, 180 + beat * 75);
            const bg = Math.min(255, 200 + energy * 55);
            const bb = Math.min(255, 220 + energy * 35);

            for (let i = 0; i < N; i++) {
                psx[i] = sx[i]; psy[i] = sy[i]; psz[i] = sz[i];
                sz[i] -= speed;

                if (sz[i] <= 0.01) {
                    sx[i] = (Math.random() - 0.5) * 2;
                    sy[i] = (Math.random() - 0.5) * 2;
                    sz[i] = 0.85 + Math.random() * 0.15;
                    psx[i] = sx[i]; psy[i] = sy[i]; psz[i] = sz[i];
                    continue;
                }

                const scrX = cx + (sx[i] / sz[i]) * fov;
                const scrY = cy + (sy[i] / sz[i]) * fov;
                const prevX = cx + (psx[i] / psz[i]) * fov;
                const prevY = cy + (psy[i] / psz[i]) * fov;

                if (scrX < -50 || scrX > w + 50 || scrY < -50 || scrY > h + 50) {
                    sx[i] = (Math.random() - 0.5) * 2;
                    sy[i] = (Math.random() - 0.5) * 2;
                    sz[i] = 0.85 + Math.random() * 0.15;
                    psx[i] = sx[i]; psy[i] = sy[i]; psz[i] = sz[i];
                    continue;
                }

                const brightness = clamp((1 - sz[i]) * 1.5, 0.1, 1);
                const cr = Math.min(255, br * brightness);
                const cg = Math.min(255, bg * brightness);
                const cb = Math.min(255, bb * brightness);

                tc.strokeStyle = rgb(cr, cg, cb);
                tc.lineWidth = 1;
                tc.beginPath();
                tc.moveTo(prevX, prevY);
                tc.lineTo(scrX, scrY);
                tc.stroke();

                tc.fillStyle = rgb(Math.min(255, cr + 50), Math.min(255, cg + 50), Math.min(255, cb + 50));
                tc.fillRect(scrX - 1, scrY - 1, 2, 2);
            }
            ctx.drawImage(trail, 0, 0);
        }
    };
}


// Aurora / Govee palette — shared between Aurora preset and Govee bridge
const PALETTE = [
    [20, 240, 100], [40, 200, 160], [30, 180, 220], [80, 120, 255], [140, 60, 220],
    [20, 255, 130], [60, 220, 180], [100, 160, 255], [160, 80, 200], [30, 250, 120]
];

// ═════════════════════════════════════════════════════════════════
// PRESET 10: AURORA
// ═════════════════════════════════════════════════════════════════
function createAurora() {
    let trail, tc;
    let time = 0;
    // Deterministic jitter to break uniform grid
    const jitter = [];
    for (let i = 0; i < 20; i++) {
        jitter[i] = Math.sin(i * 7.3 + i * i * 0.13) * 0.5;
    }

    // Cached treeline silhouette
    let treeCanvas, treeW = 0, treeH = 0;

    // Rolling terrain — composite sine waves for natural hill contour
    function terrain(x, baseY, amplitude, freqSeed) {
        return baseY
            + Math.sin(x * 0.003 + freqSeed) * amplitude * 0.5
            + Math.sin(x * 0.0071 + freqSeed * 1.7) * amplitude * 0.3
            + Math.sin(x * 0.017 + freqSeed * 3.1) * amplitude * 0.2;
    }

    // Organic spruce/pine silhouette — asymmetric branch tiers
    function drawSpruce(g, cx, baseY, treeH, treeW, rand) {
        const tipY = baseY - treeH;
        const trunkW = treeW * 0.08;
        const lean = (rand() - 0.5) * treeW * 0.12;
        const tiers = 4 + (rand() * 4) | 0;

        g.lineTo(cx - trunkW, baseY);
        // Left side ascending
        for (let t = 0; t < tiers; t++) {
            const frac = (t + 0.3) / tiers;
            const ty = baseY - treeH * frac;
            const tierLean = lean * frac;
            const spread = treeW * 0.5 * (1 - frac * 0.7) * (0.85 + rand() * 0.3);
            rand(); // consume matching random for right-side symmetry
            const droop = 3 + rand() * 4;
            g.lineTo(cx + tierLean - spread, ty + droop);
            g.lineTo(cx + tierLean - spread * 0.25, ty - rand() * 2);
        }
        // Tip — slightly off-center
        g.lineTo(cx + lean + (rand() - 0.5) * 2, tipY);
        // Right side descending
        for (let t = tiers - 1; t >= 0; t--) {
            const frac = (t + 0.3) / tiers;
            const ty = baseY - treeH * frac;
            const tierLean = lean * frac;
            const spreadR = treeW * 0.5 * (1 - frac * 0.7) * (0.85 + rand() * 0.3);
            const droop = 3 + rand() * 4;
            g.lineTo(cx + tierLean + spreadR * 0.25, ty - rand() * 2);
            g.lineTo(cx + tierLean + spreadR, ty + droop);
        }
        g.lineTo(cx + trunkW, baseY);
    }

    // Dead snag — bare trunk with 1-3 broken limbs
    function drawSnag(g, cx, baseY, treeH, rand) {
        const tipY = baseY - treeH;
        const trunkW = 2 + rand() * 2;
        g.lineTo(cx - trunkW, baseY);
        g.lineTo(cx - trunkW * 0.6, tipY + treeH * 0.1);
        g.lineTo(cx - trunkW * 0.3, tipY);
        g.lineTo(cx + trunkW * 0.2, tipY + treeH * 0.05);
        g.lineTo(cx + trunkW * 0.5, tipY + treeH * 0.08);
        const limbs = 1 + (rand() * 2) | 0;
        for (let l = 0; l < limbs; l++) {
            const ly = baseY - treeH * (0.3 + rand() * 0.4);
            const dir = rand() > 0.5 ? 1 : -1;
            const len = 6 + rand() * 10;
            g.lineTo(cx + trunkW * 0.5, ly + 2);
            g.lineTo(cx + dir * len, ly - 2 - rand() * 4);
            g.lineTo(cx + trunkW * 0.5, ly - 1);
        }
        g.lineTo(cx + trunkW, baseY);
    }

    // Dense low shrub cluster
    function drawShrubCluster(g, x, baseY, clusterW, maxH, rand) {
        const bumps = 2 + (rand() * 3) | 0;
        const bw = clusterW / bumps;
        g.lineTo(x, baseY);
        for (let b = 0; b < bumps; b++) {
            const bx = x + b * bw;
            const bh = maxH * (0.4 + rand() * 0.6);
            const peakX = bx + bw * (0.3 + rand() * 0.4);
            g.quadraticCurveTo(peakX, baseY - bh, bx + bw, baseY + rand() * 2);
        }
    }

    function buildTreeline(w, h) {
        treeW = w; treeH = h;
        treeCanvas = document.createElement('canvas');
        treeCanvas.width = w; treeCanvas.height = h;
        const g = treeCanvas.getContext('2d');

        let seed = 31417;
        const rand = () => { seed = (seed * 16807 + 0) % 2147483647; return (seed & 0x7fffffff) / 0x7fffffff; };

        // ─── 4-layer treeline with atmospheric perspective ─────────
        const layers = [
            { seed: 11213, baseF: 0.56, scale: 0.35, col: [16, 22, 30], amp: 10, tseed: 0.0 },   // distant
            { seed: 31417, baseF: 0.59, scale: 0.55, col: [10, 14, 20], amp: 14, tseed: 2.3 },   // mid-far
            { seed: 54773, baseF: 0.63, scale: 0.75, col: [6, 9, 13],  amp: 12, tseed: 4.7 },    // mid-near
            { seed: 77713, baseF: 0.67, scale: 1.0,  col: [3, 4, 6],   amp: 8,  tseed: 7.1 }     // foreground
        ];

        for (const layer of layers) {
            seed = layer.seed;
            const baseY0 = h * layer.baseF;
            const sc = layer.scale;
            const [lr, lg, lb] = layer.col;

            g.fillStyle = rgb(lr, lg, lb);
            g.beginPath();
            g.moveTo(0, h);

            // Start at left edge terrain
            const startY = terrain(0, baseY0, layer.amp, layer.tseed);
            g.lineTo(0, startY);

            let x = -5 + rand() * 10;
            while (x < w + 50) {
                const groundY = terrain(x, baseY0, layer.amp, layer.tseed);
                const r = rand();

                if (r < 0.05 && sc > 0.5) {
                    // Dead snag (rare, only in closer layers)
                    const snagH = (25 + rand() * 50) * sc;
                    drawSnag(g, x + 3, groundY, snagH, rand);
                    x += 8 + rand() * 10;
                } else if (r < 0.15) {
                    // Shrub cluster
                    const cw = (15 + rand() * 25) * sc;
                    const ch = (10 + rand() * 20) * sc;
                    drawShrubCluster(g, x, groundY, cw, ch, rand);
                    x += cw + rand() * 4 * sc;
                } else if (r < 0.22) {
                    // Gap — just follow terrain briefly (natural clearing)
                    const gapW = (8 + rand() * 20) * sc;
                    g.lineTo(x + gapW, terrain(x + gapW, baseY0, layer.amp, layer.tseed));
                    x += gapW;
                } else {
                    // Spruce/pine — the dominant tree type
                    const tall = rand() > 0.3;
                    const treeHeight = tall
                        ? (50 + rand() * 90) * sc
                        : (20 + rand() * 35) * sc;
                    const treeWidth = tall
                        ? (10 + rand() * 14) * sc
                        : (8 + rand() * 10) * sc;

                    drawSpruce(g, x + treeWidth * 0.5, groundY, treeHeight, treeWidth, rand);

                    // Clumping: sometimes pack trees tightly, sometimes leave space
                    const clump = rand();
                    if (clump < 0.3) {
                        x += treeWidth * 0.6; // tight cluster
                    } else if (clump < 0.7) {
                        x += treeWidth + rand() * 6 * sc; // normal
                    } else {
                        x += treeWidth + rand() * 18 * sc; // wider gap
                    }
                }
            }

            // Close path along bottom
            g.lineTo(w, terrain(w, baseY0, layer.amp, layer.tseed));
            g.lineTo(w, h);
            g.closePath();
            g.fill();
        }

        // Solid ground fill below foreground layer
        g.fillStyle = rgb(3, 4, 6);
        g.fillRect(0, h * 0.72, w, h * 0.28);
    }

    return {
        name: 'Aurora',
        render(ctx, audio, dt, w, h) {
            if (!trail || trail.width !== w || trail.height !== h) {
                ({ canvas: trail, ctx: tc } = makeTrail(w, h));
            }
            if (treeW !== w || treeH !== h) buildTreeline(w, h);
            time += dt;
            const da = 1 - Math.pow(1 - 28 / 255, dt * 30);
            tc.fillStyle = rgba(...BG, da);
            tc.fillRect(0, 0, w, h);

            const ribbonCount = clamp(Math.ceil(w / 120), 10, 20);
            const bpc = Math.max(1, (64 / ribbonCount) | 0);
            const brightness = 1 + audio.beatPulse * 0.3;
            const auroraBottom = h * 0.6;
            const auroraHeight = auroraBottom;
            const segments = 24;

            tc.save();
            tc.globalCompositeOperation = 'lighter';

            for (let c = 0; c < ribbonCount; c++) {
                let energy = 0;
                const lo = Math.min(c * bpc, 63), hi = Math.min(lo + bpc + 1, 64);
                for (let b = lo; b < hi; b++) energy += audio.spectrum[b];
                energy /= (hi - lo);

                const ribbonW = w * 0.12 + energy * w * 0.08;
                const baseX = (c / ribbonCount) * w + w / (2 * ribbonCount) + jitter[c % 20] * w * 0.05;
                const [cr, cg, cb] = PALETTE[c % 10];

                // Build left and right edge control points
                const leftPts = [], rightPts = [];
                for (let s = 0; s <= segments; s++) {
                    const t = s / segments;
                    const y = auroraBottom - t * auroraHeight;
                    const undulateL = Math.sin(time * 0.3 + c * 0.7 + t * 3.0) * (w * 0.02 + energy * w * 0.03);
                    const undulateR = Math.sin(time * 0.3 + c * 0.7 + t * 3.0 + 0.4) * (w * 0.02 + energy * w * 0.03);
                    leftPts.push({ x: baseX - ribbonW / 2 + undulateL, y });
                    rightPts.push({ x: baseX + ribbonW / 2 + undulateR, y });
                }

                // Vertical gradient: bright at bottom, transparent at top
                const rr = Math.min(255, cr * brightness) | 0;
                const rg = Math.min(255, cg * brightness) | 0;
                const rb = Math.min(255, cb * brightness) | 0;
                const grad = tc.createLinearGradient(0, auroraBottom, 0, 0);
                grad.addColorStop(0, rgba(rr, rg, rb, clamp(energy * 0.25, 0, 0.45)));
                grad.addColorStop(0.3, rgba(rr, rg, rb, clamp(energy * 0.12, 0, 0.25)));
                grad.addColorStop(0.7, rgba(rr, rg, rb, clamp(energy * 0.04, 0, 0.1)));
                grad.addColorStop(1.0, rgba(rr, rg, rb, 0));

                // Draw curtain as smooth closed path
                tc.beginPath();
                tc.moveTo(leftPts[0].x, leftPts[0].y);
                for (let s = 1; s <= segments; s++) {
                    const mx = (leftPts[s - 1].x + leftPts[s].x) / 2;
                    const my = (leftPts[s - 1].y + leftPts[s].y) / 2;
                    tc.quadraticCurveTo(leftPts[s - 1].x, leftPts[s - 1].y, mx, my);
                }
                tc.lineTo(leftPts[segments].x, leftPts[segments].y);
                tc.lineTo(rightPts[segments].x, rightPts[segments].y);
                for (let s = segments - 1; s >= 0; s--) {
                    const mx = (rightPts[s + 1].x + rightPts[s].x) / 2;
                    const my = (rightPts[s + 1].y + rightPts[s].y) / 2;
                    tc.quadraticCurveTo(rightPts[s + 1].x, rightPts[s + 1].y, mx, my);
                }
                tc.closePath();
                tc.fillStyle = grad;
                tc.fill();

                // Vertical rays within the curtain
                for (let r = 0; r < 3; r++) {
                    const rayPhase = time * 0.15 + c * 2.1 + r * 1.8;
                    const rayT = Math.sin(rayPhase) * 0.5 + 0.5;
                    const bottomUndulate = Math.sin(time * 0.3 + c * 0.7) * (w * 0.02 + energy * w * 0.03);
                    const rayX = baseX - ribbonW / 2 + rayT * ribbonW + bottomUndulate;
                    const topUndulate = Math.sin(time * 0.3 + c * 0.7 + 0.7 * 3.0) * (w * 0.02 + energy * w * 0.03);
                    const rayGrad = tc.createLinearGradient(0, auroraBottom, 0, auroraBottom - auroraHeight * 0.7);
                    rayGrad.addColorStop(0, rgba(Math.min(255, rr * 1.3) | 0, Math.min(255, rg * 1.3) | 0, Math.min(255, rb * 1.3) | 0, energy * 0.15));
                    rayGrad.addColorStop(1, rgba(rr, rg, rb, 0));
                    tc.strokeStyle = rayGrad;
                    tc.lineWidth = 1.5 + energy * 2;
                    tc.beginPath();
                    tc.moveTo(rayX, auroraBottom);
                    tc.lineTo(rayX + topUndulate, auroraBottom - auroraHeight * 0.7);
                    tc.stroke();
                }
            }

            tc.restore();
            ctx.drawImage(trail, 0, 0);

            // Treeline silhouette
            ctx.drawImage(treeCanvas, 0, 0);
        }
    };
}


// ═════════════════════════════════════════════════════════════════
// PRESET 11: MURMURATION
// ═════════════════════════════════════════════════════════════════
// A flock of hundreds moving as one organism. Bass pulls the flock
// together, mids align it into streams, treble scatters nervous
// energy through it, beats burst it apart. When the flock flies in
// true coherence, an eye opens at its heart and the swarm orbits it.
function createMurmuration() {
    const N = 420;
    const bx = new Float32Array(N), by = new Float32Array(N);
    const bvx = new Float32Array(N), bvy = new Float32Array(N);
    const isGold = new Uint8Array(N);
    for (let i = 0; i < N; i++) if (i % 24 === 0) isGold[i] = 1;
    let seeded = false;

    // Spatial hash grid for neighbor lookup
    const CELL = 48;
    let gridW = 0, gridH = 0, cellHead = null;
    const cellNext = new Int32Array(N);

    let t = 0;                       // attractor wander clock
    const EYE = false;               // the coherence "eye" — hidden so the flock reads alone; flip true to bring it back
    let eyeCharge = 0;               // coherence builds this toward 1
    let eyeTimer = 0;                // seconds the eye stays open
    let eyeOpen = 0;                 // eased openness 0..1
    let eyeCooldown = 0;
    let eyeX = 0, eyeY = 0;
    let trail, tc;

    return {
        name: 'Murmuration',
        render(ctx, audio, dt, w, h) {
            if (!trail || trail.width !== w || trail.height !== h) {
                ({ canvas: trail, ctx: tc } = makeTrail(w, h));
                gridW = Math.ceil(w / CELL) + 1;
                gridH = Math.ceil(h / CELL) + 1;
                cellHead = new Int32Array(gridW * gridH);
            }
            if (!seeded) {
                for (let i = 0; i < N; i++) {
                    bx[i] = w * (0.3 + Math.random() * 0.4);
                    by[i] = h * (0.3 + Math.random() * 0.4);
                    const a = Math.random() * Math.PI * 2;
                    bvx[i] = Math.cos(a) * 80; bvy[i] = Math.sin(a) * 80;
                }
                seeded = true;
            }

            const bass = audio.bass, mid = audio.mid, treble = audio.treble;
            const energy = audio.energy, beat = audio.beatPulse;

            // Silky trails, brighter decay when loud
            const da = 1 - Math.pow(1 - (15 + energy * 12) / 255, dt * 30);
            tc.fillStyle = rgba(...BG, da);
            tc.fillRect(0, 0, w, h);

            // Wandering attractor (or the eye, when open)
            t += dt * (0.5 + energy * 1.3);
            let ax = w * (0.5 + 0.22 * Math.sin(t * 0.43) * Math.cos(t * 0.19));
            let ay = h * (0.46 + 0.19 * Math.sin(t * 0.31 + 1.7));

            // Flock centroid + polarization (how aligned the flock flies)
            let cxSum = 0, cySum = 0, uxSum = 0, uySum = 0;
            for (let i = 0; i < N; i++) {
                cxSum += bx[i]; cySum += by[i];
                const sp = Math.hypot(bvx[i], bvy[i]) || 1;
                uxSum += bvx[i] / sp; uySum += bvy[i] / sp;
            }
            const cX = cxSum / N, cY = cySum / N;
            const polarization = Math.hypot(uxSum, uySum) / N;

            // ── Eye lifecycle ──
            eyeCooldown -= dt;
            if (eyeTimer <= 0) {
                eyeCharge = clamp(eyeCharge + (polarization > 0.58
                    ? dt * (0.6 + bass) : -dt * 0.4), 0, 1);
                if (EYE && eyeCharge > 0.65 && audio.beatDetected && eyeCooldown <= 0) {
                    eyeTimer = 3.2;
                    eyeCooldown = 12;
                    eyeCharge = 0;
                    eyeX = cX; eyeY = cY;
                }
                eyeOpen = Math.max(0, eyeOpen - dt * 3);
            } else {
                eyeTimer -= dt;
                const closing = Math.min(1, eyeTimer / 0.45);
                eyeOpen = Math.min(Math.min(1, eyeOpen + dt * 3.5), closing);
                ax = eyeX; ay = eyeY;
            }

            // ── Rebuild spatial grid ──
            cellHead.fill(-1);
            for (let i = 0; i < N; i++) {
                const gx = clamp((bx[i] / CELL) | 0, 0, gridW - 1);
                const gy = clamp((by[i] / CELL) | 0, 0, gridH - 1);
                const c = gy * gridW + gx;
                cellNext[i] = cellHead[c];
                cellHead[c] = i;
            }

            const kAli = 2.4 + mid * 7;
            const kCoh = 0.9 + bass * 3.4;
            const maxSpeed = 190 + energy * 260;
            const margin = 70;
            // Only heavy hits scatter the flock, and only part of it
            const scatter = audio.beatDetected && bass > 0.32;
            const scatterKick = (bass - 0.32) * 560;

            for (let i = 0; i < N; i++) {
                const x = bx[i], y = by[i];
                let sepX = 0, sepY = 0, aliX = 0, aliY = 0, cohX = 0, cohY = 0, nn = 0;

                const gx = clamp((x / CELL) | 0, 0, gridW - 1);
                const gy = clamp((y / CELL) | 0, 0, gridH - 1);
                outer:
                for (let oy = -1; oy <= 1; oy++) {
                    const yy = gy + oy;
                    if (yy < 0 || yy >= gridH) continue;
                    for (let ox = -1; ox <= 1; ox++) {
                        const xx = gx + ox;
                        if (xx < 0 || xx >= gridW) continue;
                        for (let j = cellHead[yy * gridW + xx]; j !== -1; j = cellNext[j]) {
                            if (j === i) continue;
                            const dx = bx[j] - x, dy = by[j] - y;
                            const d2 = dx * dx + dy * dy;
                            if (d2 > 2116) continue;          // 46px perception
                            const d = Math.sqrt(d2) || 0.01;
                            if (d < 20) { sepX -= dx / d * (1 - d / 20); sepY -= dy / d * (1 - d / 20); }
                            aliX += bvx[j]; aliY += bvy[j];
                            cohX += dx; cohY += dy;
                            if (++nn >= 14) break outer;
                        }
                    }
                }

                let fx = sepX * 1300, fy = sepY * 1300;
                if (nn > 0) {
                    fx += (aliX / nn - bvx[i]) * kAli + (cohX / nn) * kCoh;
                    fy += (aliY / nn - bvy[i]) * kAli + (cohY / nn) * kCoh;
                }

                // Steer toward attractor / eye ("arrive": converges, never orbits)
                let dax = ax - x, day = ay - y;
                const dad = Math.hypot(dax, day) || 1;
                const cruise = 150 + energy * 210;
                const kArr = eyeOpen > 0.1 ? 2.4 : 1.0;
                fx += (dax / dad * cruise - bvx[i]) * kArr;
                fy += (day / dad * cruise - bvy[i]) * kArr;
                // Orbit the open eye instead of piling onto it
                if (eyeOpen > 0.1 && dad < 340) {
                    fx += (-day / dad) * 220 * eyeOpen;
                    fy += (dax / dad) * 220 * eyeOpen;
                    if (dad < 130) { fx -= dax / dad * 500; fy -= day / dad * 500; }
                }

                // Edge steer
                if (x < margin) fx += (1 - x / margin) * 900;
                if (x > w - margin) fx -= (1 - (w - x) / margin) * 900;
                if (y < margin) fy += (1 - y / margin) * 900;
                if (y > h - margin) fy -= (1 - (h - y) / margin) * 900;

                // Treble nerves + beat burst away from the centroid
                if (treble > 0.05) {
                    fx += (Math.random() - 0.5) * treble * 1100;
                    fy += (Math.random() - 0.5) * treble * 1100;
                }
                if (scatter && eyeOpen < 0.1 && Math.random() < 0.35) {
                    const ddx = x - cX, ddy = y - cY;
                    const dd = Math.hypot(ddx, ddy) || 1;
                    bvx[i] += ddx / dd * scatterKick; bvy[i] += ddy / dd * scatterKick;
                }

                bvx[i] += fx * dt; bvy[i] += fy * dt;
                const sp = Math.hypot(bvx[i], bvy[i]) || 1;
                const cap = sp > maxSpeed ? maxSpeed / sp : (sp < 50 ? 50 / sp : 1);
                bvx[i] *= cap; bvy[i] *= cap;

                const px = x, py = y;
                bx[i] = x + bvx[i] * dt; by[i] = y + bvy[i] * dt;

                // Draw: speed-colored ribbon + bright head
                const sfrac = clamp((sp - 50) / (maxSpeed - 50), 0, 1);
                let cr, cg, cb;
                if (isGold[i]) { cr = 255; cg = 205 + sfrac * 30; cb = 100; }
                else {
                    cr = 120 + sfrac * -20 + beat * 60;
                    cg = 90 + sfrac * 135;
                    cb = 235 + sfrac * 20;
                }
                tc.strokeStyle = rgba(cr, cg, cb, 0.9);
                tc.lineWidth = isGold[i] ? 2.2 : 1.6;
                tc.beginPath();
                tc.moveTo(px, py);
                tc.lineTo(bx[i], by[i]);
                tc.stroke();
                tc.fillStyle = rgba(Math.min(255, cr + 70), Math.min(255, cg + 70), Math.min(255, cb + 70), 0.9);
                tc.fillRect(bx[i] - 1, by[i] - 1, 2, 2);
            }

            // ── The eye ──
            if (eyeOpen > 0.02) {
                const ew = Math.min(w, h) * 0.21 * (1 + bass * 0.12);
                const eh = ew * 0.62 * eyeOpen;
                const irisR = eh * 0.82;

                // Dark hollow behind it
                const back = tc.createRadialGradient(eyeX, eyeY, 0, eyeX, eyeY, ew * 1.5);
                back.addColorStop(0, rgba(4, 3, 8, 0.85 * eyeOpen));
                back.addColorStop(1, rgba(4, 3, 8, 0));
                tc.fillStyle = back;
                tc.fillRect(eyeX - ew * 1.5, eyeY - ew * 1.5, ew * 3, ew * 3);

                // Almond lids as clip
                tc.save();
                tc.beginPath();
                tc.moveTo(eyeX - ew, eyeY);
                tc.quadraticCurveTo(eyeX, eyeY - eh * 1.7, eyeX + ew, eyeY);
                tc.quadraticCurveTo(eyeX, eyeY + eh * 1.7, eyeX - ew, eyeY);
                tc.closePath();
                tc.clip();

                // Iris
                const iris = tc.createRadialGradient(eyeX, eyeY, 0, eyeX, eyeY, irisR);
                iris.addColorStop(0, rgb(8, 6, 12));
                iris.addColorStop(0.28 + audio.sub * 0.2, rgb(10, 8, 14));
                iris.addColorStop(0.45, rgb(180, 120, 40));
                iris.addColorStop(0.75, rgb(255, 200, 90));
                iris.addColorStop(1, rgb(90, 60, 150));
                tc.fillStyle = iris;
                tc.beginPath();
                tc.arc(eyeX, eyeY, irisR, 0, Math.PI * 2);
                tc.fill();

                // Radial striations
                tc.strokeStyle = rgba(255, 220, 140, 0.25 * eyeOpen);
                tc.lineWidth = 1;
                for (let s = 0; s < 24; s++) {
                    const a = (s / 24) * Math.PI * 2 + t * 0.1;
                    tc.beginPath();
                    tc.moveTo(eyeX + Math.cos(a) * irisR * 0.4, eyeY + Math.sin(a) * irisR * 0.4);
                    tc.lineTo(eyeX + Math.cos(a) * irisR * 0.95, eyeY + Math.sin(a) * irisR * 0.95);
                    tc.stroke();
                }

                // Glint
                tc.fillStyle = rgba(255, 255, 240, 0.8 * eyeOpen);
                tc.beginPath();
                tc.arc(eyeX - irisR * 0.28, eyeY - irisR * 0.3, irisR * 0.09, 0, Math.PI * 2);
                tc.fill();
                tc.restore();

                // Lid edges
                tc.strokeStyle = rgba(220, 200, 255, 0.7 * eyeOpen);
                tc.lineWidth = 1.5;
                tc.beginPath();
                tc.moveTo(eyeX - ew, eyeY);
                tc.quadraticCurveTo(eyeX, eyeY - eh * 1.7, eyeX + ew, eyeY);
                tc.quadraticCurveTo(eyeX, eyeY + eh * 1.7, eyeX - ew, eyeY);
                tc.closePath();
                tc.stroke();
            }

            ctx.drawImage(trail, 0, 0);
        }
    };
}


// ═════════════════════════════════════════════════════════════════
// PRESET 12: VORTEX
// ═════════════════════════════════════════════════════════════════
// The namesake. Two reciprocal logarithmic spirals — one gold, one
// cyan, opposite chirality, counter-rotating — carry the 64 bands
// from sub-bass at the rim to treble at the core. Particles ride
// the arms down into a single point of light that rings on beats.
function createVortex() {
    const THETA_MAX = Math.PI * 5;   // 2.5 turns per arm
    const STEPS = 200;
    const NP = 240;                  // infall particles
    const pTheta = new Float32Array(NP);
    const pChir = new Uint8Array(NP);
    const pArm = new Uint8Array(NP);
    for (let i = 0; i < NP; i++) {
        pTheta[i] = Math.random() * THETA_MAX;
        pChir[i] = Math.random() < 0.5 ? 0 : 1;
        pArm[i] = Math.random() < 0.5 ? 0 : 1;
    }
    const rings = [];                // beat pulses expanding outward
    let rot = 0;
    let trail, tc;

    // Band amplitude along the arm: treble at the core, sub at the rim
    function armAmp(spectrum, frac) {
        const fi = (1 - frac) * 63;
        const i0 = fi | 0, i1 = Math.min(63, i0 + 1);
        const ft = fi - i0;
        return spectrum[i0] * (1 - ft) + spectrum[i1] * ft;
    }

    return {
        name: 'Vortex',
        render(ctx, audio, dt, w, h) {
            if (!trail || trail.width !== w || trail.height !== h) {
                ({ canvas: trail, ctx: tc } = makeTrail(w, h));
            }
            const energy = audio.energy, bass = audio.bass, beat = audio.beatPulse;
            const da = 1 - Math.pow(1 - (65 + energy * 20) / 255, dt * 30);
            tc.fillStyle = rgba(...BG, da);
            tc.fillRect(0, 0, w, h);

            const cx = w / 2, cy = h / 2;
            const rOut = Math.min(w, h) * 0.46;
            const rCore = 9;
            const k = Math.log(rOut / rCore) / THETA_MAX;
            rot += dt * (0.12 + energy * 0.7 + beat * 0.3);

            // ── Infall particles — the spectrum drains into the 1 ──
            const infall = dt * (0.45 + bass * 2.4 + beat * 1.1);
            for (let i = 0; i < NP; i++) {
                pTheta[i] -= infall * (0.6 + 0.9 * (1 - pTheta[i] / THETA_MAX));
                if (pTheta[i] < 0.04) {
                    pTheta[i] = THETA_MAX * (0.9 + Math.random() * 0.1);
                    pChir[i] = Math.random() < 0.5 ? 0 : 1;
                    pArm[i] = Math.random() < 0.5 ? 0 : 1;
                    continue;
                }
                const th = pTheta[i];
                const sgn = pChir[i] === 0 ? 1 : -1;
                const phase = sgn * rot * (pChir[i] === 0 ? 1 : 0.82);
                const frac = th / THETA_MAX;
                const amp = armAmp(audio.spectrum, frac);
                const r = rCore * Math.exp(k * th) * (1 + amp * 0.18);
                const a = sgn * th + phase + pArm[i] * Math.PI;
                const x = cx + Math.cos(a) * r, y = cy + Math.sin(a) * r;
                const [cr, cg, cb] = SPECTRUM_COLORS[Math.min(63, ((1 - frac) * 63) | 0)];
                const bright = 0.6 + amp * 0.4;
                tc.fillStyle = rgba(cr * bright + 80, cg * bright + 80, cb * bright + 80, 0.95);
                const sz = 2.2 + amp * 2 + (1 - frac) * 1.6;
                tc.fillRect(x - sz / 2, y - sz / 2, sz, sz);
            }

            // ── Beat rings — the pulse of the core travels outward ──
            if (audio.beatDetected) rings.push({ r: rCore + 4, a: 0.55 });
            for (let i = rings.length - 1; i >= 0; i--) {
                const ring = rings[i];
                ring.r += (240 + bass * 420) * dt;
                ring.a -= dt * 0.55;
                if (ring.a <= 0 || ring.r > rOut * 1.2) { rings.splice(i, 1); continue; }
                tc.strokeStyle = rgba(255, 235, 190, ring.a);
                tc.lineWidth = 1.5;
                tc.beginPath();
                tc.arc(cx, cy, ring.r, 0, Math.PI * 2);
                tc.stroke();
            }

            ctx.drawImage(trail, 0, 0);

            // ── Arms — drawn crisp every frame: 2 chiralities × 2 arms ──
            for (let chir = 0; chir < 2; chir++) {
                const sgn = chir === 0 ? 1 : -1;
                const phase = sgn * rot * (chir === 0 ? 1 : 0.82);
                for (let arm = 0; arm < 2; arm++) {
                    const off = arm * Math.PI;

                    // Wide soft under-stroke for glow
                    ctx.strokeStyle = chir === 0
                        ? rgba(255, 170, 60, 0.15 + energy * 0.10 + beat * 0.06)
                        : rgba(80, 140, 255, 0.15 + energy * 0.10 + beat * 0.06);
                    ctx.lineWidth = 5;
                    ctx.lineCap = 'round';
                    ctx.beginPath();
                    for (let s = 2; s <= STEPS - 6; s += 2) {
                        const th = (s / STEPS) * THETA_MAX;
                        const amp = armAmp(audio.spectrum, th / THETA_MAX);
                        const r = rCore * Math.exp(k * th) * (1 + amp * 0.18);
                        const a = sgn * th + phase + off;
                        const x = cx + Math.cos(a) * r, y = cy + Math.sin(a) * r;
                        s === 2 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
                    }
                    ctx.stroke();

                    // Bright pass — segmented, amplitude-lit, tapered tips
                    for (let s = 0; s < STEPS; s += 4) {
                        const th0 = (s / STEPS) * THETA_MAX;
                        const th1 = ((s + 4) / STEPS) * THETA_MAX;
                        const frac = th0 / THETA_MAX;
                        const amp = armAmp(audio.spectrum, frac);
                        const taper = frac > 0.9 ? (1 - frac) / 0.1 : 1;
                        const lum = (0.42 + amp * 0.58) * taper;
                        const r0 = rCore * Math.exp(k * th0) * (1 + amp * 0.18);
                        const r1 = rCore * Math.exp(k * th1) * (1 + armAmp(audio.spectrum, th1 / THETA_MAX) * 0.18);
                        const a0 = sgn * th0 + phase + off, a1 = sgn * th1 + phase + off;
                        ctx.strokeStyle = chir === 0
                            ? rgba(255 * lum, (185 + beat * 50) * lum, 90 * lum, 0.9)
                            : rgba(110 * lum, (190 + beat * 45) * lum, 255 * lum, 0.9);
                        ctx.lineWidth = 1.3 + amp * 1.8;
                        ctx.beginPath();
                        ctx.moveTo(cx + Math.cos(a0) * r0, cy + Math.sin(a0) * r0);
                        ctx.lineTo(cx + Math.cos(a1) * r1, cy + Math.sin(a1) * r1);
                        ctx.stroke();
                    }
                }
            }

            // ── The core — one point of light, drawn crisp each frame ──
            const coreR = 7 + energy * 24 + beat * 26;
            const core = ctx.createRadialGradient(cx, cy, 0, cx, cy, coreR * 2.4);
            core.addColorStop(0, rgba(255, 255, 245, 0.95));
            core.addColorStop(0.25, rgba(255, 225, 150, 0.55 + beat * 0.3));
            core.addColorStop(0.6, rgba(180, 140, 255, 0.18));
            core.addColorStop(1, rgba(180, 140, 255, 0));
            ctx.fillStyle = core;
            ctx.beginPath();
            ctx.arc(cx, cy, coreR * 2.4, 0, Math.PI * 2);
            ctx.fill();
        }
    };
}


// ═════════════════════════════════════════════════════════════════
// PRESET 13: CYMATICS
// ═════════════════════════════════════════════════════════════════
// A Chladni plate. Thousands of sand grains migrate to the nodal
// lines of a standing wave whose mode numbers follow the music —
// bass picks the coarse pattern, treble the fine one. Beats strike
// the plate and shatter the figure; quiet lets it crystallize.
function createCymatics() {
    const N = 3600;
    const gu = new Float32Array(N), gv = new Float32Array(N);
    let seeded = false;

    // Every clean, iconic square-plate Chladni figure the plate can ring,
    // ordered simple → intricate (√(m²+n²) ascending). Each figure is a
    // single mode, so it stays crisp; the plate sweeps through the whole
    // set on a timer, morphing smoothly from one figure into the next.
    const LADDER = [
        { m: 1, n: 2 }, { m: 1, n: 3 }, { m: 2, n: 3 }, { m: 1, n: 4 },
        { m: 2, n: 4 }, { m: 3, n: 4 }, { m: 1, n: 5 }, { m: 2, n: 5 },
        { m: 3, n: 5 }, { m: 2, n: 6 }, { m: 4, n: 5 }, { m: 3, n: 6 },
        { m: 4, n: 6 }, { m: 3, n: 7 }, { m: 5, n: 6 }, { m: 4, n: 7 },
        { m: 5, n: 7 }, { m: 6, n: 7 }, { m: 5, n: 8 }, { m: 7, n: 8 },
    ];
    const K = LADDER.length;
    const SWITCH_EVERY = 10;             // seconds on each figure before it shifts
    const MORPH_TIME = 2.0;              // seconds to tween from one figure into the next

    let rungFrom = 0, rungTo = 0;        // figure we're morphing from / to
    let dir = 1;                         // sweep direction (bounces at the ends)
    let sweepT = 0;                      // seconds elapsed on the current figure
    let morph = 1;                       // 0 = just switched, 1 = fully settled on rungTo
    let strike = 0;                      // beat mallet

    // Settled grains glow warm sand; airborne grains stay cool dust
    const WARM = [], COOL = [];
    for (let i = 0; i < 10; i++) {
        const s = i / 9;
        WARM.push(rgba(255, 208 + s * 30, 130 + s * 50, 0.35 + s * 0.55));
        COOL.push(rgba(120, 140, 185, 0.16 + s * 0.2));
    }

    return {
        name: 'Cymatics',
        render(ctx, audio, dt, w, h) {
            if (!seeded) {
                for (let i = 0; i < N; i++) {
                    gu[i] = Math.random() * 2 - 1;
                    gv[i] = Math.random() * 2 - 1;
                }
                seeded = true;
            }
            ctx.fillStyle = rgb(...BG);
            ctx.fillRect(0, 0, w, h);

            const bass = audio.bass, treble = audio.treble, energy = audio.energy;

            // ── Which figure? Sweep the whole ladder on a slow timer — a new
            //    clean figure every SWITCH_EVERY seconds, bouncing at the ends.
            //    We count playing time, so it holds while the music is paused.
            if (energy > 0.02) sweepT += dt;
            if (sweepT >= SWITCH_EVERY && K > 1) {
                sweepT = 0;
                let nx = rungTo + dir;
                if (nx >= K) { nx = K - 2; dir = -1; }
                else if (nx < 0) { nx = 1; dir = 1; }
                rungFrom = rungTo;
                rungTo = nx;
                morph = 0;                                    // begin the tween into the new figure
            }
            // Ease the tween (smoothstep hurries through the busy middle) and
            // blend the two figures' fields — the nodal lines flow from one
            // pattern into the next, so grains glide across instead of cutting.
            morph = Math.min(1, morph + dt / MORPH_TIME);
            const wght = morph * morph * (3 - 2 * morph);     // 0..1 blend weight: rungFrom → rungTo
            const morphing = wght < 0.9995;
            const A = LADDER[rungFrom], B = LADDER[rungTo];
            const MpiB = B.m * Math.PI, NpiB = B.n * Math.PI;
            const MpiA = A.m * Math.PI, NpiA = A.n * Math.PI;
            const wB = wght, wA = 1 - wght;

            // Beat = mallet strike (a shimmer of energy through the sand)
            if (audio.beatDetected) strike = Math.min(1, strike + 0.1 + Math.max(0, bass - 0.3) * 1.3);
            strike *= Math.pow(0.14, dt);

            const side = Math.min(w, h) * 0.84;
            const half = side / 2;
            const cx = w / 2, cy = h / 2;

            // Plate chrome
            ctx.strokeStyle = rgba(150, 160, 190, 0.16);
            ctx.lineWidth = 1;
            ctx.strokeRect(cx - half, cy - half, side, side);
            const tick = 7;
            ctx.strokeStyle = rgba(150, 160, 190, 0.35);
            for (const [tx, ty] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
                ctx.beginPath();
                ctx.moveTo(cx + tx * half, cy + ty * half - ty * tick);
                ctx.lineTo(cx + tx * half, cy + ty * half);
                ctx.lineTo(cx + tx * half - tx * tick, cy + ty * half);
                ctx.stroke();
            }

            const jit = 0.0035 + treble * 0.010 + energy * 0.005 + strike * 0.05;
            const G = 2.3 * dt;
            const MAXSTEP = 0.035;

            for (let i = 0; i < N; i++) {
                const u = gu[i], v = gv[i];

                // Destination figure φ_B and its gradient (always needed)
                const cMuB = Math.cos(MpiB * u), sMuB = Math.sin(MpiB * u);
                const cNuB = Math.cos(NpiB * u), sNuB = Math.sin(NpiB * u);
                const cMvB = Math.cos(MpiB * v), sMvB = Math.sin(MpiB * v);
                const cNvB = Math.cos(NpiB * v), sNvB = Math.sin(NpiB * v);
                let f    = cMuB * cNvB - cNuB * cMvB;
                let dfdu = -MpiB * sMuB * cNvB + NpiB * sNuB * cMvB;
                let dfdv = -NpiB * cMuB * sNvB + MpiB * cNuB * sMvB;

                // Mid-tween, blend in the figure we're leaving so the field
                // (1−w)·φ_A + w·φ_B deforms continuously and the grains flow
                if (morphing) {
                    f *= wB; dfdu *= wB; dfdv *= wB;
                    const cMuA = Math.cos(MpiA * u), sMuA = Math.sin(MpiA * u);
                    const cNuA = Math.cos(NpiA * u), sNuA = Math.sin(NpiA * u);
                    const cMvA = Math.cos(MpiA * v), sMvA = Math.sin(MpiA * v);
                    const cNvA = Math.cos(NpiA * v), sNvA = Math.sin(NpiA * v);
                    f    += wA * (cMuA * cNvA - cNuA * cMvA);
                    dfdu += wA * (-MpiA * sMuA * cNvA + NpiA * sNuA * cMvA);
                    dfdv += wA * (-NpiA * cMuA * sNvA + MpiA * cNuA * sMvA);
                }

                // Drift toward nodal lines (φ = 0) + thermal shake
                let du = -f * dfdu * G + (Math.random() - 0.5) * jit;
                let dv = -f * dfdv * G + (Math.random() - 0.5) * jit;
                if (du > MAXSTEP) du = MAXSTEP; else if (du < -MAXSTEP) du = -MAXSTEP;
                if (dv > MAXSTEP) dv = MAXSTEP; else if (dv < -MAXSTEP) dv = -MAXSTEP;

                let nu = u + du, nv = v + dv;
                if (nu > 1) nu = 2 - nu; else if (nu < -1) nu = -2 - nu;
                if (nv > 1) nv = 2 - nv; else if (nv < -1) nv = -2 - nv;
                gu[i] = nu; gv[i] = nv;

                // Settled grains glow warm; airborne grains cool and dim
                const sett = 1 - Math.min(1, Math.abs(f) * 1.5);
                const lvl = (sett * 9.99) | 0;
                ctx.fillStyle = sett > 0.55 ? WARM[lvl] : COOL[lvl];
                const px = cx + nu * half, py = cy + nv * half;
                const sz = sett > 0.55 ? 2 : 1.5;
                ctx.fillRect(px - sz / 2, py - sz / 2, sz, sz);
            }

            // Driver at plate center — sub-bass + beat, swelling through a tween
            const morphGlow = morphing ? Math.sin(Math.PI * morph) : 0;
            const drvR = 16 + audio.sub * 46 + strike * 30 + morphGlow * 20;
            const drv = ctx.createRadialGradient(cx, cy, 0, cx, cy, drvR);
            drv.addColorStop(0, rgba(255, 214, 140, 0.10 + audio.sub * 0.22 + strike * 0.15 + morphGlow * 0.10));
            drv.addColorStop(1, rgba(255, 214, 140, 0));
            ctx.fillStyle = drv;
            ctx.beginPath();
            ctx.arc(cx, cy, drvR, 0, Math.PI * 2);
            ctx.fill();

            // Readout — names the figure, or the tween in progress
            ctx.font = '11px "Courier New", monospace';
            ctx.fillStyle = rgba(150, 160, 190, 0.5);
            ctx.textAlign = 'left';
            const label = morphing ? `mode (${A.m},${A.n}) → (${B.m},${B.n})` : `mode (${B.m},${B.n})`;
            ctx.fillText(label, cx - half + 2, cy - half - 8);
        }
    };
}


function createSupportScreen() {
    let hoverBtn = false;
    const BMC_URL = 'https://buymeacoffee.com/joeyv23';
    const particles = [];
    for (let i = 0; i < 60; i++) {
        particles.push({
            x: Math.random(), y: Math.random(),
            vx: (Math.random() - 0.5) * 0.02,
            vy: (Math.random() - 0.5) * 0.02,
            size: 1 + Math.random() * 2,
            alpha: 0.1 + Math.random() * 0.3,
            hue: 180 + Math.random() * 60
        });
    }

    const screen = {
        name: 'Support',
        isSupportScreen: true,
        btnRect: null,

        render(ctx, audio, dt, w, h) {
            // Background
            ctx.fillStyle = rgb(...BG);
            ctx.fillRect(0, 0, w, h);

            // Ambient floating particles
            for (const p of particles) {
                p.x += p.vx * dt;
                p.y += p.vy * dt;
                if (p.x < 0 || p.x > 1) p.vx *= -1;
                if (p.y < 0 || p.y > 1) p.vy *= -1;
                ctx.beginPath();
                ctx.arc(p.x * w, p.y * h, p.size, 0, Math.PI * 2);
                ctx.fillStyle = `hsla(${p.hue}, 80%, 60%, ${p.alpha})`;
                ctx.fill();
            }

            // Respond to audio if playing
            const energy = audio.energy || 0;
            const glowAlpha = 0.05 + energy * 0.15;

            // Radial glow behind content
            const grad = ctx.createRadialGradient(w / 2, h * 0.38, 0, w / 2, h * 0.38, w * 0.35);
            grad.addColorStop(0, `rgba(80, 200, 255, ${glowAlpha})`);
            grad.addColorStop(1, 'transparent');
            ctx.fillStyle = grad;
            ctx.fillRect(0, 0, w, h);

            const compact = w < 500;
            const cx = w / 2;

            // Vortex icon (simple spiral)
            const iconY = h * 0.22;
            const iconR = compact ? 28 : 40;
            ctx.save();
            ctx.strokeStyle = 'rgba(80, 200, 255, 0.7)';
            ctx.lineWidth = 2;
            ctx.beginPath();
            for (let a = 0; a < Math.PI * 6; a += 0.1) {
                const r = iconR * (a / (Math.PI * 6));
                const x = cx + Math.cos(a) * r;
                const y = iconY + Math.sin(a) * r;
                a === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            }
            ctx.stroke();
            ctx.restore();

            // Title
            const titleSize = compact ? 20 : 32;
            ctx.font = `bold ${titleSize}px "Courier New", monospace`;
            ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
            ctx.textAlign = 'center';
            ctx.fillText('Enjoying Vortex?', cx, iconY + iconR + (compact ? 30 : 45));

            // Subtitle
            const subSize = compact ? 12 : 16;
            ctx.font = `${subSize}px "Courier New", monospace`;
            ctx.fillStyle = 'rgba(180, 220, 255, 0.7)';
            const subY = iconY + iconR + (compact ? 50 : 75);
            ctx.fillText('Vortex is free and open source.', cx, subY);
            ctx.fillText('If you dig it, consider buying me a coffee.', cx, subY + subSize + 6);

            // Button
            const btnW = compact ? 200 : 260;
            const btnH = compact ? 44 : 52;
            const btnX = cx - btnW / 2;
            const btnY = subY + (compact ? 40 : 60);
            const btnR = 8;

            // Button glow on hover
            if (hoverBtn) {
                ctx.shadowColor = 'rgba(255, 221, 0, 0.4)';
                ctx.shadowBlur = 20;
            }

            // Button background — BMC yellow
            ctx.beginPath();
            ctx.roundRect(btnX, btnY, btnW, btnH, btnR);
            ctx.fillStyle = hoverBtn ? '#ffe940' : '#ffdd00';
            ctx.fill();
            ctx.shadowBlur = 0;

            // Coffee cup icon (simple)
            const cupX = btnX + (compact ? 18 : 24);
            const cupY = btnY + btnH / 2;
            const cupS = compact ? 8 : 10;
            ctx.fillStyle = '#0d0d0d';
            ctx.fillRect(cupX - cupS / 2, cupY - cupS / 2, cupS, cupS * 1.1);
            ctx.strokeStyle = '#0d0d0d';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(cupX + cupS / 2 + 2, cupY, cupS * 0.4, -Math.PI / 2, Math.PI / 2);
            ctx.stroke();
            // Steam
            ctx.lineWidth = 1;
            for (let i = 0; i < 3; i++) {
                const sx = cupX - cupS / 4 + i * (cupS / 2.5);
                ctx.beginPath();
                ctx.moveTo(sx, cupY - cupS / 2 - 2);
                ctx.quadraticCurveTo(sx + 2, cupY - cupS / 2 - 6, sx, cupY - cupS / 2 - 10);
                ctx.stroke();
            }

            // Button text
            const btnFontSize = compact ? 14 : 17;
            ctx.font = `bold ${btnFontSize}px "Courier New", monospace`;
            ctx.fillStyle = '#0d0d0d';
            ctx.textAlign = 'center';
            ctx.fillText('Buy Me a Coffee', cx + (compact ? 6 : 8), btnY + btnH / 2 + btnFontSize / 3);

            screen.btnRect = { x: btnX, y: btnY, w: btnW, h: btnH };

            // Footer
            const footSize = compact ? 10 : 12;
            ctx.font = `${footSize}px "Courier New", monospace`;
            ctx.fillStyle = 'rgba(140, 160, 180, 0.5)';
            ctx.textAlign = 'center';
            ctx.fillText('← swipe to return to visualizers →', cx, h - (compact ? 30 : 40));

            ctx.textAlign = 'left';
        },

        handleClick(x, y) {
            const r = screen.btnRect;
            if (r && x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h) {
                window.open(BMC_URL, '_blank');
                return true;
            }
            return false;
        },

        handleMove(x, y) {
            const r = screen.btnRect;
            hoverBtn = r && x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h;
        }
    };
    return screen;
}


// ═════════════════════════════════════════════════════════════════
// HUD / UI
// ═════════════════════════════════════════════════════════════════
class UI {
    constructor() {
        this.hideTimer = 5;
        this.toasts = [];
    }

    activity() { this.hideTimer = 5; }

    toast(text, duration) { this.toasts.push({ text, time: duration || 1.5 }); }

    render(ctx, engine, presetName, dt, w, h) {
        // Toast notifications (always visible)
        for (let i = this.toasts.length - 1; i >= 0; i--) {
            const t = this.toasts[i];
            t.time -= dt;
            if (t.time <= 0) { this.toasts.splice(i, 1); continue; }
            ctx.save();
            ctx.globalAlpha = t.time < 0.5 ? t.time / 0.5 : 1;
            const fontSize = w < 500 ? 13 : 22;
            const lineH = fontSize + 6;
            ctx.font = `${fontSize}px "Courier New", monospace`;
            // Word-wrap toast text to fit screen
            const maxW = w - 40;
            const words = t.text.split(' ');
            const lines = [];
            let line = '';
            for (const word of words) {
                const test = line ? line + ' ' + word : word;
                if (ctx.measureText(test).width > maxW && line) {
                    lines.push(line);
                    line = word;
                } else {
                    line = test;
                }
            }
            if (line) lines.push(line);
            // Background box
            const boxH = lines.length * lineH + 16;
            const boxW = Math.min(maxW + 24, w - 16);
            ctx.fillStyle = 'rgba(0,0,0,0.75)';
            ctx.fillRect((w - boxW) / 2, 24, boxW, boxH);
            // Text
            ctx.textAlign = 'center';
            ctx.fillStyle = '#dcdce6';
            for (let j = 0; j < lines.length; j++) {
                ctx.fillText(lines[j], w / 2, 46 + j * lineH);
            }
            ctx.restore();
        }

        // HUD bar
        this.hideTimer -= dt;
        if (this.hideTimer <= 0) return;
        const alpha = this.hideTimer < 1 ? this.hideTimer : 1;

        ctx.save();
        ctx.globalAlpha = alpha;

        const hudH = 48;
        ctx.fillStyle = 'rgba(0,0,0,0.6)';
        ctx.fillRect(0, h - hudH, w, hudH);

        ctx.font = '14px "Courier New", monospace';

        if (engine.isCapturing) {
            // Live capture HUD — pulsing red dot + mode label
            const pulse = 0.6 + 0.4 * Math.sin(Date.now() / 300);
            ctx.fillStyle = `rgba(255, 60, 60, ${pulse})`;
            ctx.beginPath();
            ctx.arc(18, h - hudH + 18, 5, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = '#dcdce6';
            const modeLabel = { 'live-system': 'System Audio', 'live-display': 'Screen Audio', 'live-mic': 'Microphone' };
            ctx.fillText(`LIVE: ${modeLabel[engine.mode] || 'Capture'}`, 30, h - hudH + 20);
        } else {
            ctx.fillStyle = '#dcdce6';
            const status = engine.playing ? '\u25B6' : '\u23F8';
            const name = engine.trackName || 'No track';
            const num = engine.trackCount > 0 ? `[${engine.currentTrack + 1}/${engine.trackCount}]` : '';
            const timeStr = `${formatTime(engine.currentTime)} / ${formatTime(engine.duration)}`;
            ctx.fillText(`${status}  ${name}  ${num}  ${timeStr}`, 12, h - hudH + 20);
        }

        // Preset name
        ctx.textAlign = 'right';
        ctx.fillStyle = '#50c8ff';
        ctx.fillText(presetName, w - 12, h - hudH + 20);

        // Volume + modes
        let modeStr = `Vol ${Math.round(engine.volume * 100)}%`;
        if (engine.shuffle) modeStr += '  [S]';
        if (engine.repeat) modeStr += '  [R]';
        ctx.fillStyle = '#78788c';
        ctx.fillText(modeStr, w - 12, h - hudH + 38);
        ctx.textAlign = 'left';

        ctx.restore();
    }
}


// ═════════════════════════════════════════════════════════════════
// PRESET 14: COSMOS — "Godfellas" galactic entity (audio-reactive)
//   A face-on spiral of physically-coloured stars (blackbody by
//   temperature, mostly cool dwarfs) on deep space. The bright
//   "speaking" stars pulse with the spectrum, and every beat launches
//   a wave of brightening that sweeps outward across the galaxy.
// ═════════════════════════════════════════════════════════════════
function createCosmos() {
    const TAU = Math.PI * 2;
    let time = 0, angle = 0, sEnergy = 0;

    // speech-wave ripples (spawned on each beat, travel outward)
    const ripples = [];
    const RIPPLE_SPEED = 1.5, RIPPLE_MAX = 4, RIPPLE_LIFE = 1.3, RIPPLE_W = 0.15;

    // ── helpers ──────────────────────────────────────────────────
    const lerp = (a, b, t) => a + (b - a) * t;
    const lerpCol = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
    const gauss = () => (Math.random() + Math.random() + Math.random() - 1.5) / 1.5;

    // blackbody temperature (K) → sRGB from Charity's table, with a mild
    // saturation lift so O reads blue / M reads orange on screen
    const BB = [
        [3000, 255, 180, 107], [3500, 255, 196, 137], [4000, 255, 209, 163], [4500, 255, 219, 186],
        [5000, 255, 228, 206], [5500, 255, 236, 224], [6600, 255, 249, 253], [7000, 245, 243, 255],
        [8000, 227, 233, 255], [10000, 204, 219, 255], [15000, 179, 204, 255], [20000, 168, 196, 255],
        [30000, 159, 190, 255]
    ];
    function bbColor(t) {
        let c;
        if (t <= BB[0][0]) c = [BB[0][1], BB[0][2], BB[0][3]];
        else if (t >= BB[BB.length - 1][0]) { const l = BB[BB.length - 1]; c = [l[1], l[2], l[3]]; }
        else {
            for (let i = 1; i < BB.length; i++) {
                if (t <= BB[i][0]) {
                    const a = BB[i - 1], b = BB[i], f = (t - a[0]) / (b[0] - a[0]);
                    c = [lerp(a[1], b[1], f), lerp(a[2], b[2], f), lerp(a[3], b[3], f)];
                    break;
                }
            }
        }
        const lum = 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2];
        return [clamp(lum + (c[0] - lum) * 1.4, 0, 255), clamp(lum + (c[1] - lum) * 1.4, 0, 255), clamp(lum + (c[2] - lum) * 1.4, 0, 255)];
    }
    function sampleTemp() {
        const r = Math.random();
        if (r < 0.72) return 3000 + Math.random() * 2100;   // M / K  (majority)
        if (r < 0.90) return 5200 + Math.random() * 1800;   // G / F
        if (r < 0.98) return 7000 + Math.random() * 4000;   // A / late B
        return 12000 + Math.random() * 16000;               // rare O / B
    }

    // the reactive "speaking" stars — real bright-star colours, warm→cool,
    // index-matched to ascending spectrum band (red supergiants … blue giants)
    const SPEAK_COL = [
        [255, 180, 120], [255, 186, 128], [255, 201, 152], [255, 214, 176], [255, 225, 190], [255, 239, 228],
        [255, 246, 237], [248, 247, 255], [214, 225, 255], [204, 219, 255], [201, 218, 255], [178, 203, 255],
        [168, 196, 255], [159, 190, 255]
    ];
    const NSPEAK = SPEAK_COL.length;

    function mkStar() {
        const b = Math.pow(Math.random(), 2.2);
        return {
            size: 0.55 + b * 1.9, b, col: bbColor(sampleTemp()),
            twPhase: Math.random() * TAU, twFreq: 0.4 + Math.random() * 1.6,
            twAmp: 0.12 + Math.random() * 0.28
        };
    }

    // ── build the field once ─────────────────────────────────────
    const SWEEP = 2.5;
    const field = [];                       // diffuse backdrop (screen-frac, static)
    for (let i = 0; i < 500; i++) {
        const s = mkStar(); s.fx = Math.random(); s.fy = Math.random(); field.push(s);
    }
    const galaxy = [];                      // spiral arms + bulge (centered, rotates)
    for (let i = 0; i < 340; i++) {
        const arm = i % 2, t = Math.pow(Math.random(), 0.8);
        const r = 0.13 + t * 1.02 + (Math.random() - 0.5) * 0.13;
        const th = arm * Math.PI + t * SWEEP + (Math.random() - 0.5) * 0.5;
        const s = mkStar();
        s.gx = Math.cos(th) * r; s.gy = Math.sin(th) * r; s.dist = Math.hypot(s.gx, s.gy);
        galaxy.push(s);
    }
    for (let i = 0; i < 130; i++) {
        const s = mkStar();
        s.gx = gauss() * 0.24; s.gy = gauss() * 0.24; s.dist = Math.hypot(s.gx, s.gy);
        s.b = Math.max(s.b, 0.3 + Math.random() * 0.4);
        galaxy.push(s);
    }
    const speak = [];                       // reactive "speaking" stars
    for (let i = 0; i < NSPEAK; i++) {
        let gx = 0, gy = 0;
        if (i > 0) {
            const th = (i / NSPEAK) * TAU + (Math.random() - 0.5) * 0.6, r = 0.16 + Math.random() * 0.98;
            gx = Math.cos(th) * r; gy = Math.sin(th) * r;
        }
        speak.push({
            gx, gy, dist: Math.hypot(gx, gy),
            band: Math.min(63, 2 + Math.round(i / (NSPEAK - 1) * 60)),
            col: SPEAK_COL[i], baseSize: 3 + Math.random() * 2, lit: 0.2
        });
    }
    const nebula = [];                      // soft gas blobs (centered, rotates)
    //  warm-white bulge core · then Hα-red / [O III]-teal / reflection-blue arms
    nebula.push({ gx: 0, gy: 0, rad: 0.5, core: true, col: [255, 240, 224] });
    for (let i = 0; i < 12; i++) {
        const arm = i % 2, t = 0.15 + i / 12, th = arm * Math.PI + t * SWEEP, r = (0.13 + t * 0.95) * 0.85;
        const role = i % 5;
        nebula.push({
            gx: Math.cos(th) * r, gy: Math.sin(th) * r,
            rad: 0.18 + Math.random() * 0.14, core: false,
            col: role === 0 ? [255, 70, 80] : role === 3 ? [0, 205, 175] : [80, 110, 205]
        });
    }

    // ── draw one glowing star (soft halo + diffraction spikes + hot core) ──
    function star(ctx, x, y, col, lit, base) {
        const R = base * (0.8 + lit * 0.8);
        const g = ctx.createRadialGradient(x, y, 0, x, y, R * 4);
        g.addColorStop(0, rgba(col[0], col[1], col[2], clamp(0.34 * lit + 0.06, 0, 1)));
        g.addColorStop(1, rgba(col[0], col[1], col[2], 0));
        ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, R * 4, 0, TAU); ctx.fill();
        if (lit > 0.15) {
            const spike = R * (2 + lit * 7), sa = clamp(lit * 0.5, 0, 0.6);
            const line = (x1, y1, x2, y2) => {
                const lg = ctx.createLinearGradient(x1, y1, x2, y2);
                lg.addColorStop(0, rgba(col[0], col[1], col[2], 0));
                lg.addColorStop(0.5, rgba(255, 255, 255, sa));
                lg.addColorStop(1, rgba(col[0], col[1], col[2], 0));
                ctx.strokeStyle = lg; ctx.lineWidth = 1.1;
                ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
            };
            line(x - spike, y, x + spike, y); line(x, y - spike, x, y + spike);
        }
        const cc = lerpCol(col, [255, 255, 255], clamp(lit * 0.7, 0, 0.8));
        ctx.fillStyle = rgba(cc[0], cc[1], cc[2], clamp(0.55 + lit * 0.45, 0, 1));
        ctx.beginPath(); ctx.arc(x, y, R * 0.62, 0, TAU); ctx.fill();
    }

    return {
        name: 'Cosmos',
        render(ctx, audio, dt, w, h) {
            dt = Math.min(dt, 0.05);
            time += dt; angle += dt * 0.012;
            sEnergy += (audio.energy - sEnergy) * (1 - Math.pow(0.6, dt * 30));
            const energy = sEnergy, bass = audio.bass, mid = audio.mid;

            if (audio.beatDetected && ripples.length < RIPPLE_MAX) ripples.push({ t: 0 });
            for (let i = ripples.length - 1; i >= 0; i--) {
                ripples[i].t += dt;
                if (ripples[i].t > RIPPLE_LIFE) ripples.splice(i, 1);
            }
            const rippleAt = (d) => {
                let s = 0;
                for (const rp of ripples) {
                    const x = d - rp.t * RIPPLE_SPEED;
                    s += Math.exp(-(x * x) / (2 * RIPPLE_W * RIPPLE_W)) * (1 - rp.t / RIPPLE_LIFE);
                }
                return s;
            };

            const cx = w / 2, cy = h / 2, S = Math.min(w, h) * 0.46;
            const ca = Math.cos(angle), sa = Math.sin(angle);

            ctx.globalCompositeOperation = 'source-over'; ctx.globalAlpha = 1;
            ctx.fillStyle = rgb(...BG); ctx.fillRect(0, 0, w, h);
            ctx.globalCompositeOperation = 'lighter';

            for (const b of nebula) {
                const rx = b.gx * ca - b.gy * sa, ry = b.gx * sa + b.gy * ca;
                const x = cx + rx * S, y = cy + ry * S;
                const breath = b.core ? (0.85 + bass * 0.6) : (0.8 + mid * 0.4 + energy * 0.2);
                const rad = b.rad * S * (0.9 + energy * 0.2);
                const a = clamp((b.core ? 0.16 : 0.10) * breath, 0, 0.4);
                const g = ctx.createRadialGradient(x, y, 0, x, y, rad);
                g.addColorStop(0, rgba(b.col[0], b.col[1], b.col[2], a));
                g.addColorStop(1, rgba(b.col[0], b.col[1], b.col[2], 0));
                ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, rad, 0, TAU); ctx.fill();
            }

            const gLift = 0.85 + energy * 0.35;
            for (const s of field) {
                const tw = 1 + s.twAmp * Math.sin(time * s.twFreq + s.twPhase);
                const a = clamp(s.b * tw * gLift * 0.9, 0, 1), sz = s.size * (0.9 + 0.15 * tw);
                ctx.fillStyle = rgba(s.col[0], s.col[1], s.col[2], a);
                ctx.beginPath(); ctx.arc(s.fx * w, s.fy * h, sz, 0, TAU); ctx.fill();
                if (s.b > 0.55) {
                    ctx.fillStyle = rgba(s.col[0], s.col[1], s.col[2], a * 0.16);
                    ctx.beginPath(); ctx.arc(s.fx * w, s.fy * h, sz * 2.4, 0, TAU); ctx.fill();
                }
            }

            for (const s of galaxy) {
                const rx = s.gx * ca - s.gy * sa, ry = s.gx * sa + s.gy * ca;
                const x = cx + rx * S, y = cy + ry * S;
                const tw = 1 + s.twAmp * Math.sin(time * s.twFreq + s.twPhase);
                const rip = rippleAt(s.dist) * 0.5;
                const a = clamp((s.b * tw * gLift + rip) * 0.95, 0, 1), sz = s.size * (0.9 + 0.15 * tw + rip * 0.5);
                ctx.fillStyle = rgba(s.col[0], s.col[1], s.col[2], a);
                ctx.beginPath(); ctx.arc(x, y, sz, 0, TAU); ctx.fill();
                if (s.b > 0.5 || rip > 0.2) {
                    ctx.fillStyle = rgba(s.col[0], s.col[1], s.col[2], a * 0.18);
                    ctx.beginPath(); ctx.arc(x, y, sz * 2.4, 0, TAU); ctx.fill();
                }
            }

            for (const s of speak) {
                const rx = s.gx * ca - s.gy * sa, ry = s.gx * sa + s.gy * ca;
                const x = cx + rx * S, y = cy + ry * S;
                const be = audio.spectrum[s.band] || 0, rip = rippleAt(s.dist);
                const target = 0.18 + be * 0.95 + rip * 0.9 + audio.beatPulse * 0.12;
                s.lit += (target - s.lit) * (1 - Math.pow(0.5, dt * 30));
                star(ctx, x, y, s.col, clamp(s.lit, 0, 1.5), s.baseSize);
            }

            ctx.globalCompositeOperation = 'source-over'; ctx.globalAlpha = 1;
        }
    };
}


// ═════════════════════════════════════════════════════════════════
// APP — Main Loop & Event Handling
// ═════════════════════════════════════════════════════════════════
class App {
    constructor() {
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.audio = new AudioEngine();
        this.ui = new UI();
        this.presets = [
            createSpectrumBars(),
            createOscilloscope(),
            createParticleFountain(),
            createCircularWaveform(),
            createRadialSpectrum(),
            createSpectrogram(),
            createLissajous(),
            createSacredGeometry(),
            createStarfield(),
            createAurora(),
            createMurmuration(),
            createVortex(),
            createCymatics(),
            createCosmos(),
            createSupportScreen()
        ];
        this.currentPreset = 0;
        this.lastTime = 0;
        this.w = 0;
        this.h = 0;

        // Crossfade
        this.prevPreset = -1;
        this.transitionTime = 0;
        this.transitionDuration = 0.6;
        this.offA = document.createElement('canvas');
        this.offB = document.createElement('canvas');
        this.ctxA = this.offA.getContext('2d');
        this.ctxB = this.offB.getContext('2d');

        // Auto-cycle
        this.autoCycle = false;
        this.autoCycleTimer = 0;
        this.autoCycleInterval = 10;

        // Panels
        this.helpEl = document.getElementById('help-overlay');
        this.playlistEl = document.getElementById('playlist-panel');
        this.plTracksEl = document.getElementById('pl-tracks');

        // Mobile state
        this.isMobile = 'ontouchstart' in window || (navigator.maxTouchPoints > 0);
        this.mobileVisible = false;
        this.mobileHideTimer = 0;

        this._resize();
        this._setupEvents();
        if (this.isMobile) this._setupMobile();
        this.audio.onTrackChange = () => { this._updatePlaylist(); this._updateMobilePlayBtn(); };
        this._loop = this._loop.bind(this);
        requestAnimationFrame(this._loop);

        // Govee Aurora Bridge — optional WebSocket to sync lights.
        // Local-dev only: the bridge server runs on the operator's machine,
        // so visitors at the public URL would hit a permanent reconnect loop
        // against their own localhost. Gate to localhost / 127.0.0.1.
        this._goveeWs = null;
        this._goveeConnecting = false;
        const isLocal = ['localhost', '127.0.0.1', '0.0.0.0'].includes(location.hostname);
        if (isLocal) this._connectGoveeBridge();
    }

    _connectGoveeBridge() {
        if (this._goveeConnecting) return;
        this._goveeConnecting = true;
        try {
            const ws = new WebSocket('ws://localhost:9876');
            ws.onopen = () => {
                this._goveeWs = ws;
                this._goveeConnecting = false;
                this.ui.toast('Govee bridge connected');
            };
            ws.onclose = () => {
                this._goveeWs = null;
                this._goveeConnecting = false;
                setTimeout(() => this._connectGoveeBridge(), 5000);
            };
            ws.onerror = () => {
                ws.close();
            };
        } catch (e) {
            this._goveeConnecting = false;
            setTimeout(() => this._connectGoveeBridge(), 5000);
        }
    }

    _resize() {
        this.w = window.innerWidth;
        this.h = window.innerHeight;
        this.canvas.width = this.w;
        this.canvas.height = this.h;
        this.offA.width = this.w; this.offA.height = this.h;
        this.offB.width = this.w; this.offB.height = this.h;
    }

    _setupEvents() {
        window.addEventListener('resize', () => this._resize());
        window.addEventListener('keydown', (e) => this._handleKey(e));

        // Drag & drop
        const dz = document.getElementById('drop-zone');
        let dragCount = 0;
        window.addEventListener('dragenter', (e) => { e.preventDefault(); dragCount++; dz.classList.add('active'); });
        window.addEventListener('dragleave', (e) => { e.preventDefault(); if (--dragCount <= 0) { dragCount = 0; dz.classList.remove('active'); } });
        window.addEventListener('dragover', (e) => e.preventDefault());
        window.addEventListener('drop', async (e) => {
            e.preventDefault();
            dragCount = 0;
            dz.classList.remove('active');
            await this._loadFiles(e.dataTransfer.files);
        });

        // Browse / Add files buttons — all trigger the same file input
        // AudioContext init moved to _loadFiles (change handler) so fi.click()
        // stays synchronous with the user gesture — Firefox blocks it otherwise
        const fi = document.getElementById('file-input');
        const openPicker = () => {
            fi.click();
        };
        document.getElementById('browse-btn').addEventListener('click', openPicker);
        document.getElementById('pl-add-btn').addEventListener('click', openPicker);
        fi.addEventListener('change', async () => { await this._loadFiles(fi.files); fi.value = ''; });

        this.canvas.addEventListener('mousemove', (e) => {
            this.ui.activity();
            const preset = this.presets[this.currentPreset];
            if (preset.handleMove) {
                const rect = this.canvas.getBoundingClientRect();
                preset.handleMove(e.clientX - rect.left, e.clientY - rect.top);
                this.canvas.style.cursor = preset.isSupportScreen && preset.btnRect &&
                    e.clientX - rect.left >= preset.btnRect.x && e.clientX - rect.left <= preset.btnRect.x + preset.btnRect.w &&
                    e.clientY - rect.top >= preset.btnRect.y && e.clientY - rect.top <= preset.btnRect.y + preset.btnRect.h
                    ? 'pointer' : '';
            } else {
                this.canvas.style.cursor = '';
            }
        });
        this.canvas.addEventListener('click', (e) => {
            const preset = this.presets[this.currentPreset];
            if (preset.handleClick) {
                const rect = this.canvas.getBoundingClientRect();
                preset.handleClick(e.clientX - rect.left, e.clientY - rect.top);
            }
        });

        // Live audio picker
        this._setupLivePicker();

        // Desktop seek bar
        this._setupDesktopSeek();

        // Playlist click
        this.plTracksEl.addEventListener('click', (e) => {
            const el = e.target.closest('.pl-track');
            if (el && el.dataset.idx !== undefined) {
                this.audio.play(parseInt(el.dataset.idx));
            }
        });
    }

    async _loadFiles(files) {
        if (!this.audio.actx) await this.audio.init();
        const before = this.audio.trackCount;
        this.audio.loadFiles(files);
        const added = this.audio.trackCount - before;
        document.getElementById('welcome').style.display = 'none';
        if (before === 0) this.ui.toast(this.audio.trackName);
        else if (added > 0) this.ui.toast(`+${added} track${added > 1 ? 's' : ''} added`);
        this._updatePlaylist();
        if (this.isMobile) this._showMobileControls();
    }

    _setupLivePicker() {
        const picker = document.getElementById('live-picker');
        const caps = AudioEngine.detectCapabilities();

        // Disable unavailable options
        if (!caps.displayCapture) document.getElementById('lp-display').disabled = true;
        if (!caps.micCapture) document.getElementById('lp-mic').disabled = true;
        if (!caps.systemCapture) document.getElementById('lp-system').disabled = true;

        document.getElementById('live-btn').addEventListener('click', () => {
            picker.classList.add('active');
        });

        document.getElementById('lp-cancel').addEventListener('click', () => {
            picker.classList.remove('active');
        });

        // Callback when live capture ends externally (browser stop-sharing, etc.)
        this.audio.onLiveCaptureEnd = () => {
            this.ui.toast('Live capture ended');
            this._onLiveStopped();
        };

        document.getElementById('lp-display').addEventListener('click', async () => {
            picker.classList.remove('active');
            try {
                // Check Firefox before any async work
                if (navigator.userAgent.includes('Firefox')) {
                    this.ui.toast('Screen audio not supported in Firefox. Use Chrome/Edge or try Microphone.', 4);
                    return;
                }
                // Request display media FIRST — must stay in user gesture stack
                const stream = await navigator.mediaDevices.getDisplayMedia({
                    video: true,
                    audio: true,
                    preferCurrentTab: false,
                    systemAudio: 'include'
                });
                stream.getVideoTracks().forEach(t => t.stop());
                const audioTracks = stream.getAudioTracks();
                if (audioTracks.length === 0) {
                    stream.getTracks().forEach(t => t.stop());
                    this.ui.toast('No audio track. Check "Share audio" in the share dialog.', 4);
                    return;
                }
                if (!this.audio.actx) await this.audio.init();
                const audioStream = new MediaStream(audioTracks);
                audioTracks[0].addEventListener('ended', () => {
                    this.audio.stopLiveCapture();
                    if (this.audio.onLiveCaptureEnd) this.audio.onLiveCaptureEnd();
                });
                await this.audio.startLiveCapture(audioStream, 'live-display');
                this._onLiveStarted();
            } catch (e) {
                console.warn('[Vortex] Display capture error:', e);
                this.ui.toast(e.message || 'Capture failed', 4);
            }
        });

        document.getElementById('lp-mic').addEventListener('click', async () => {
            picker.classList.remove('active');
            try {
                // Request mic FIRST — must be in the user gesture call stack
                // before any async work, or mobile Chrome blocks it
                const stream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        echoCancellation: false,
                        noiseSuppression: false,
                        autoGainControl: false
                    }
                });
                if (!this.audio.actx) await this.audio.init();
                stream.getAudioTracks()[0].addEventListener('ended', () => {
                    this.audio.stopLiveCapture();
                    if (this.audio.onLiveCaptureEnd) this.audio.onLiveCaptureEnd();
                });
                await this.audio.startLiveCapture(stream, 'live-mic');
                this.audio.gainNode.gain.value = 0;
                this._onLiveStarted();
            } catch (e) {
                console.warn('[Vortex] Mic capture error:', e);
                this.ui.toast(e.message || 'Mic access denied', 4);
            }
        });

        document.getElementById('lp-system').addEventListener('click', async () => {
            try {
                await this.audio.startSystemCapture();
                this._onLiveStarted();
            } catch (e) {
                this.ui.toast(e.message || 'System capture failed', 4);
            }
            picker.classList.remove('active');
        });
    }

    _onLiveStarted() {
        document.getElementById('live-picker').classList.remove('active');
        document.getElementById('welcome').style.display = 'none';
        this.ui.activity();
        const modeLabel = { 'live-system': 'System Audio', 'live-display': 'Screen Audio', 'live-mic': 'Microphone' };
        this.ui.toast(`LIVE: ${modeLabel[this.audio.mode] || 'Capture'}`);
        if (this.isMobile) this._showMobileControls();
    }

    _onLiveStopped() {
        this.ui.activity();
    }

    _stopLiveCapture() {
        if (this.audio.isCapturing) {
            this.audio.stopLiveCapture();
            this.ui.toast('Live capture stopped');
            this._onLiveStopped();
        }
    }

    _switchPreset(idx) {
        if (idx === this.currentPreset) return;
        this.prevPreset = this.currentPreset;
        this.currentPreset = idx;
        this.transitionTime = 0;
        this.ui.toast(this.presets[idx].name);
    }

    _updatePlaylist() {
        const tracks = this.audio.tracks;
        const current = this.audio.currentTrack;
        if (tracks.length === 0) {
            this.plTracksEl.innerHTML = '<div class="pl-empty">No tracks loaded</div>';
            return;
        }
        const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
        let html = '';
        for (let i = 0; i < tracks.length; i++) {
            const cls = i === current ? 'pl-track active' : 'pl-track';
            html += `<div class="${cls}" data-idx="${i}">${i + 1}. ${esc(tracks[i].name)}</div>`;
        }
        this.plTracksEl.innerHTML = html;
        const active = this.plTracksEl.querySelector('.pl-track.active');
        if (active) active.scrollIntoView({ block: 'nearest' });
    }

    _setupMobile() {
        const mc = document.getElementById('mobile-controls');
        const seekBar = document.getElementById('mobile-seek');
        this.mcEl = mc;
        this.seekEl = seekBar;
        this.seekFill = document.getElementById('seek-fill');

        // Swipe gesture on canvas
        let touchStartX = 0, touchStartY = 0, touchStartTime = 0;
        this.canvas.addEventListener('touchstart', (e) => {
            touchStartX = e.touches[0].clientX;
            touchStartY = e.touches[0].clientY;
            touchStartTime = Date.now();
        }, { passive: true });
        this.canvas.addEventListener('touchend', (e) => {
            const dx = e.changedTouches[0].clientX - touchStartX;
            const dy = e.changedTouches[0].clientY - touchStartY;
            const elapsed = Date.now() - touchStartTime;

            if (Math.abs(dx) > 60 && Math.abs(dx) > Math.abs(dy) * 1.5 && elapsed < 400) {
                // Horizontal swipe → change preset
                if (dx > 0) this._switchPreset((this.currentPreset - 1 + this.presets.length) % this.presets.length);
                else this._switchPreset((this.currentPreset + 1) % this.presets.length);
            } else if (Math.abs(dx) < 20 && Math.abs(dy) < 20 && elapsed < 300) {
                // Tap — check support screen button first, then toggle controls
                const preset = this.presets[this.currentPreset];
                const tx = e.changedTouches[0].clientX;
                const ty = e.changedTouches[0].clientY;
                const rect = this.canvas.getBoundingClientRect();
                if (preset.handleClick && preset.handleClick(tx - rect.left, ty - rect.top)) {
                    // Button was tapped, don't toggle controls
                } else if (this.mobileVisible) this._hideMobileControls();
                else this._showMobileControls();
            }
        }, { passive: true });

        // Mobile seek bar touch
        const seekTouch = (e) => {
            e.preventDefault();
            const rect = seekBar.getBoundingClientRect();
            const frac = clamp((e.touches[0].clientX - rect.left) / rect.width, 0, 1);
            this.audio.seek(frac);
        };
        seekBar.addEventListener('touchstart', seekTouch, { passive: false });
        seekBar.addEventListener('touchmove', seekTouch, { passive: false });

        // Control buttons
        document.getElementById('mc-play').addEventListener('click', async () => {
            if (this.audio.actx) this.audio.toggle();
            this._updateMobilePlayBtn();
            this._resetMobileHideTimer();
        });
        document.getElementById('mc-prev').addEventListener('click', () => {
            this.audio.prevTrack();
            this._resetMobileHideTimer();
        });
        document.getElementById('mc-next').addEventListener('click', () => {
            this.audio.nextTrack();
            this._resetMobileHideTimer();
        });
        document.getElementById('mc-preset-prev').addEventListener('click', () => {
            this._switchPreset((this.currentPreset - 1 + this.presets.length) % this.presets.length);
            this._resetMobileHideTimer();
        });
        document.getElementById('mc-preset-next').addEventListener('click', () => {
            this._switchPreset((this.currentPreset + 1) % this.presets.length);
            this._resetMobileHideTimer();
        });
        document.getElementById('mc-vol-down').addEventListener('click', () => {
            this.audio.setVolume(this.audio.volume - 0.1);
            this.ui.toast(`Volume ${Math.round(this.audio.volume * 100)}%`);
            this._resetMobileHideTimer();
        });
        document.getElementById('mc-vol-up').addEventListener('click', () => {
            this.audio.setVolume(this.audio.volume + 0.1);
            this.ui.toast(`Volume ${Math.round(this.audio.volume * 100)}%`);
            this._resetMobileHideTimer();
        });
        document.getElementById('mc-add').addEventListener('click', () => {
            document.getElementById('file-input').click();
            this._resetMobileHideTimer();
        });
        document.getElementById('mc-playlist').addEventListener('click', () => {
            this.playlistEl.classList.toggle('active');
            this._resetMobileHideTimer();
        });
        document.getElementById('mc-fullscreen').addEventListener('click', () => {
            document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
            this._resetMobileHideTimer();
        });
    }

    _showMobileControls() {
        this.mobileVisible = true;
        this.mobileHideTimer = 6;
        this.mcEl.classList.add('active');
        this.seekEl.classList.add('active');
        this.ui.activity();
    }

    _hideMobileControls() {
        this.mobileVisible = false;
        this.mcEl.classList.remove('active');
        this.seekEl.classList.remove('active');
    }

    _resetMobileHideTimer() {
        this.mobileHideTimer = 6;
        this.ui.activity();
    }

    _updateMobilePlayBtn() {
        const btn = document.getElementById('mc-play');
        if (btn) btn.innerHTML = this.audio.playing ? '&#9208;' : '&#9654;';
    }

    _updateMobileSeek() {
        if (!this.seekFill || !this.mobileVisible) return;
        const pct = this.audio.duration > 0 ? (this.audio.currentTime / this.audio.duration) * 100 : 0;
        this.seekFill.style.width = pct + '%';
    }

    _setupDesktopSeek() {
        const bar = document.getElementById('desktop-seek');
        if (!bar) return;
        this.dsBar = bar;
        this.dsFill = document.getElementById('ds-fill');
        this.dsHandle = document.getElementById('ds-handle');
        this.dsTooltip = document.getElementById('ds-tooltip');
        this.dsDragging = false;

        const getFrac = (e) => clamp(e.clientX / window.innerWidth, 0, 1);

        bar.addEventListener('mousedown', (e) => {
            e.preventDefault();
            this.dsDragging = true;
            bar.classList.add('dragging');
            this.audio.seek(getFrac(e));
            this.ui.activity();
        });

        window.addEventListener('mousemove', (e) => {
            if (this.dsDragging) {
                this.audio.seek(getFrac(e));
                this.ui.activity();
            }
            // Tooltip position
            if (bar.matches(':hover') || this.dsDragging) {
                const frac = clamp(e.clientX / window.innerWidth, 0, 1);
                const dur = this.audio.duration || 0;
                this.dsTooltip.textContent = formatTime(frac * dur);
                this.dsTooltip.style.left = e.clientX + 'px';
                this.dsHandle.style.left = (frac * 100) + '%';
            }
        });

        window.addEventListener('mouseup', () => {
            if (this.dsDragging) {
                this.dsDragging = false;
                bar.classList.remove('dragging');
            }
        });

        bar.addEventListener('mousemove', (e) => {
            const frac = clamp(e.clientX / window.innerWidth, 0, 1);
            const dur = this.audio.duration || 0;
            this.dsTooltip.textContent = formatTime(frac * dur);
            this.dsTooltip.style.left = e.clientX + 'px';
            this.dsHandle.style.left = (frac * 100) + '%';
        });
    }

    _updateDesktopSeek() {
        if (!this.dsBar) return;
        const visible = this.ui.hideTimer > 0 && this.audio.duration > 0;
        this.dsBar.classList.toggle('visible', visible);
        if (visible && !this.dsDragging) {
            const pct = (this.audio.currentTime / this.audio.duration) * 100;
            this.dsFill.style.width = pct + '%';
            this.dsHandle.style.left = pct + '%';
        }
    }

    _handleKey(e) {
        this.ui.activity();
        switch (e.key) {
            case ' ':
                e.preventDefault();
                if (this.audio.actx) this.audio.toggle();
                break;
            case 'ArrowRight':
                this._switchPreset((this.currentPreset + 1) % this.presets.length);
                break;
            case 'ArrowLeft':
                this._switchPreset((this.currentPreset - 1 + this.presets.length) % this.presets.length);
                break;
            case 'ArrowUp':
                e.preventDefault();
                this.audio.setVolume(this.audio.volume + 0.05);
                this.ui.toast(`Volume ${Math.round(this.audio.volume * 100)}%`);
                break;
            case 'ArrowDown':
                e.preventDefault();
                this.audio.setVolume(this.audio.volume - 0.05);
                this.ui.toast(`Volume ${Math.round(this.audio.volume * 100)}%`);
                break;
            case 'f': case 'F':
                document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
                break;
            case 's': case 'S':
                this.audio.shuffle = !this.audio.shuffle;
                this.ui.toast(`Shuffle ${this.audio.shuffle ? 'ON' : 'OFF'}`);
                break;
            case 'r': case 'R':
                this.audio.repeat = !this.audio.repeat;
                this.ui.toast(`Repeat ${this.audio.repeat ? 'ON' : 'OFF'}`);
                break;
            case 'n': case 'N':
                this.audio.nextTrack();
                if (this.audio.trackName) this.ui.toast(this.audio.trackName);
                break;
            case 'p': case 'P':
                this.audio.prevTrack();
                if (this.audio.trackName) this.ui.toast(this.audio.trackName);
                break;
            case 'a': case 'A':
                this.autoCycle = !this.autoCycle;
                this.autoCycleTimer = 0;
                this.ui.toast(`Auto-Cycle ${this.autoCycle ? 'ON' : 'OFF'}`);
                break;
            case 'h': case 'H': case '?':
                this.helpEl.classList.toggle('active');
                break;
            case 'l': case 'L':
                this.playlistEl.classList.toggle('active');
                break;
            case 'c': case 'C':
                if (this.audio.isCapturing) this._stopLiveCapture();
                else document.getElementById('live-picker').classList.add('active');
                break;
            case 'Escape':
                this.helpEl.classList.remove('active');
                this.playlistEl.classList.remove('active');
                document.getElementById('live-picker').classList.remove('active');
                this._stopLiveCapture();
                break;
        }
        // Number keys → preset select (1–9 = preset 1–9, 0 = preset 10)
        if (e.key >= '1' && e.key <= '9') {
            const idx = parseInt(e.key) - 1;
            if (idx < this.presets.length) this._switchPreset(idx);
        }
        if (e.key === '0' && this.presets.length >= 10) {
            this._switchPreset(9);
        }
    }

    _loop(timestamp) {
        const dt = this.lastTime ? Math.min((timestamp - this.lastTime) / 1000, 0.1) : 1 / 60;
        this.lastTime = timestamp;

        const audioData = this.audio.update(dt);

        // Auto-cycle on beat
        this.autoCycleTimer += dt;
        if (this.autoCycle && audioData.beatDetected && this.autoCycleTimer >= this.autoCycleInterval) {
            this.autoCycleTimer = 0;
            let next;
            do { next = Math.floor(Math.random() * this.presets.length); } while ((next === this.currentPreset || this.presets[next].isSupportScreen) && this.presets.length > 1);
            this._switchPreset(next);
        }

        // Render with crossfade
        const preset = this.presets[this.currentPreset];
        if (this.prevPreset >= 0) {
            const t = Math.min(this.transitionTime / this.transitionDuration, 1);
            const old = this.presets[this.prevPreset];

            this.ctxA.save();
            old.render(this.ctxA, audioData, dt, this.w, this.h);
            this.ctxA.restore();

            this.ctxB.save();
            preset.render(this.ctxB, audioData, dt, this.w, this.h);
            this.ctxB.restore();

            this.ctx.fillStyle = rgb(...BG);
            this.ctx.fillRect(0, 0, this.w, this.h);
            this.ctx.globalAlpha = 1 - t;
            this.ctx.drawImage(this.offA, 0, 0);
            this.ctx.globalAlpha = t;
            this.ctx.drawImage(this.offB, 0, 0);
            this.ctx.globalAlpha = 1;

            this.transitionTime += dt;
            if (t >= 1) this.prevPreset = -1;
        } else {
            this.ctx.save();
            preset.render(this.ctx, audioData, dt, this.w, this.h);
            this.ctx.restore();
        }

        this.ui.render(this.ctx, this.audio, preset.name, dt, this.w, this.h);

        // Broadcast aurora state to Govee bridge
        if (this._goveeWs && this._goveeWs.readyState === 1 && preset.name === 'Aurora') {
            const rc = 10, bpc = Math.max(1, (64 / rc) | 0);
            const curtains = [];
            for (let c = 0; c < rc; c++) {
                let e = 0;
                const lo = Math.min(c * bpc, 63), hi = Math.min(lo + bpc + 1, 64);
                for (let b = lo; b < hi; b++) e += audioData.spectrum[b];
                e /= (hi - lo);
                curtains.push({ r: PALETTE[c][0], g: PALETTE[c][1], b: PALETTE[c][2], energy: e });
            }
            this._goveeWs.send(JSON.stringify({
                type: 'aurora', curtains,
                beatPulse: audioData.beatPulse,
                beatDetected: audioData.beatDetected,
                energy: audioData.energy
            }));
        }

        // Mobile: auto-hide controls, update seek bar
        if (this.isMobile && this.mobileVisible) {
            this.mobileHideTimer -= dt;
            if (this.mobileHideTimer <= 0) this._hideMobileControls();
            this._updateMobileSeek();
            this._updateMobilePlayBtn();
        }

        // Desktop seek bar
        this._updateDesktopSeek();

        requestAnimationFrame(this._loop);
    }
}

// ─── Launch ──────────────────────────────────────────────────────
const app = new App();
