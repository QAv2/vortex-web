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
    }

    _computeBandEdges() {
        const edges = [];
        const ratio = FREQ_MAX / FREQ_MIN;
        const binHz = 44100 / FFT_SIZE;
        for (let i = 0; i <= NUM_BANDS; i++) {
            const freq = FREQ_MIN * Math.pow(ratio, i / NUM_BANDS);
            edges.push(Math.round(freq / binHz));
        }
        return edges;
    }

    async init() {
        if (this.actx) return;
        this.actx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 44100 });
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
        for (const file of files) {
            if (file.type.startsWith('audio/') || /\.(mp3|wav|ogg|flac|m4a|aac|opus|webm)$/i.test(file.name)) {
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

    get playing() { return this.audio && !this.audio.paused; }
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

            // Beat flash
            if (audio.beatPulse > 0.01) {
                ctx.fillStyle = rgba(80, 200, 255, audio.beatPulse * 0.12);
                ctx.fillRect(0, 0, w, h);
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


// ═════════════════════════════════════════════════════════════════
// PRESET 10: AURORA
// ═════════════════════════════════════════════════════════════════
function createAurora() {
    let trail, tc;
    let time = 0;
    const PALETTE = [
        [20, 240, 100], [40, 200, 160], [30, 180, 220], [80, 120, 255], [140, 60, 220],
        [20, 255, 130], [60, 220, 180], [100, 160, 255], [160, 80, 200], [30, 250, 120]
    ];

    return {
        name: 'Aurora',
        render(ctx, audio, dt, w, h) {
            if (!trail || trail.width !== w || trail.height !== h) {
                ({ canvas: trail, ctx: tc } = makeTrail(w, h));
            }
            time += dt;
            const da = 1 - Math.pow(1 - 28 / 255, dt * 30);
            tc.fillStyle = rgba(...BG, da);
            tc.fillRect(0, 0, w, h);

            const brightness = 1 + audio.beatPulse * 0.3;
            const bpc = (64 / 10) | 0;

            for (let c = 0; c < 10; c++) {
                let energy = 0;
                const lo = c * bpc, hi = Math.min(lo + bpc + 1, 64);
                for (let b = lo; b < hi; b++) energy += audio.spectrum[b];
                energy /= (hi - lo);

                const cw = 40 + energy * 80;
                const ch = h * (0.35 + energy * 0.55);
                const baseX = (c / 10) * w + w / 20;
                const [cr, cg, cb] = PALETTE[c];
                const segments = 8;

                for (let s = 0; s < segments; s++) {
                    const t = s / segments;
                    const undulation = Math.sin(time * 0.35 + c * 0.9 + t * 2.5) * (20 + energy * 30);
                    const x = baseX + undulation;
                    const y = t * ch;
                    const segH = ch / segments;
                    const falloff = 1 - t * t;
                    const alpha = clamp(energy * 0.2 * brightness * falloff, 0, 0.6);

                    tc.fillStyle = rgba(
                        Math.min(255, cr * brightness),
                        Math.min(255, cg * brightness),
                        Math.min(255, cb * brightness),
                        alpha
                    );
                    tc.fillRect(x - cw / 2, y, cw, segH + 1);
                }
            }
            ctx.drawImage(trail, 0, 0);
        }
    };
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

    toast(text) { this.toasts.push({ text, time: 1.5 }); }

    render(ctx, engine, presetName, dt, w, h) {
        // Toast notifications (always visible)
        for (let i = this.toasts.length - 1; i >= 0; i--) {
            const t = this.toasts[i];
            t.time -= dt;
            if (t.time <= 0) { this.toasts.splice(i, 1); continue; }
            ctx.save();
            ctx.globalAlpha = t.time < 0.5 ? t.time / 0.5 : 1;
            ctx.font = '24px "Courier New", monospace';
            ctx.textAlign = 'center';
            ctx.fillStyle = '#dcdce6';
            ctx.fillText(t.text, w / 2, 50);
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
        ctx.fillStyle = '#dcdce6';
        const status = engine.playing ? '\u25B6' : '\u23F8';
        const name = engine.trackName || 'No track';
        const num = engine.trackCount > 0 ? `[${engine.currentTrack + 1}/${engine.trackCount}]` : '';
        const timeStr = `${formatTime(engine.currentTime)} / ${formatTime(engine.duration)}`;
        ctx.fillText(`${status}  ${name}  ${num}  ${timeStr}`, 12, h - hudH + 20);

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

        // Progress bar
        ctx.fillStyle = '#282832';
        ctx.fillRect(0, h - 3, w, 3);
        if (engine.duration > 0) {
            ctx.fillStyle = '#50c8ff';
            ctx.fillRect(0, h - 3, w * (engine.currentTime / engine.duration), 3);
        }

        ctx.restore();
    }
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
            createAurora()
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
        const fi = document.getElementById('file-input');
        document.getElementById('browse-btn').addEventListener('click', () => fi.click());
        document.getElementById('pl-add-btn').addEventListener('click', () => fi.click());
        fi.addEventListener('change', async () => { await this._loadFiles(fi.files); fi.value = ''; });

        // Click to seek (bottom 20px)
        this.canvas.addEventListener('click', (e) => {
            this.ui.activity();
            if (e.clientY > this.h - 20) this.audio.seek(e.clientX / this.w);
        });
        this.canvas.addEventListener('mousemove', () => this.ui.activity());

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
        let html = '';
        for (let i = 0; i < tracks.length; i++) {
            const cls = i === current ? 'pl-track active' : 'pl-track';
            html += `<div class="${cls}" data-idx="${i}">${i + 1}. ${tracks[i].name}</div>`;
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
                // Tap → toggle mobile controls
                if (this.mobileVisible) this._hideMobileControls();
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
            case 'Escape':
                this.helpEl.classList.remove('active');
                this.playlistEl.classList.remove('active');
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
            do { next = Math.floor(Math.random() * this.presets.length); } while (next === this.currentPreset && this.presets.length > 1);
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

        // Mobile: auto-hide controls, update seek bar
        if (this.isMobile && this.mobileVisible) {
            this.mobileHideTimer -= dt;
            if (this.mobileHideTimer <= 0) this._hideMobileControls();
            this._updateMobileSeek();
            this._updateMobilePlayBtn();
        }

        requestAnimationFrame(this._loop);
    }
}

// ─── Launch ──────────────────────────────────────────────────────
const app = new App();
