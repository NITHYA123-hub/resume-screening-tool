// ============================================================
// AI Resume Screening Tool - JavaScript Animations & Effects
// ============================================================

// ─── Particle Background ───────────────────────────────────
class ParticleSystem {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.particles = [];
        this.resize();
        window.addEventListener('resize', () => this.resize());
        this.init();
        this.animate();
    }

    resize() {
        this.canvas.width  = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }

    init() {
        this.particles = [];
        const count = Math.floor((this.canvas.width * this.canvas.height) / 12000);
        for (let i = 0; i < count; i++) {
            this.particles.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                vx: (Math.random() - 0.5) * 0.4,
                vy: (Math.random() - 0.5) * 0.4,
                size: Math.random() * 2 + 0.5,
                opacity: Math.random() * 0.4 + 0.1,
                color: Math.random() > 0.5 ? '79,142,247' : '139,92,246'
            });
        }
    }

    drawConnections() {
        for (let i = 0; i < this.particles.length; i++) {
            for (let j = i + 1; j < this.particles.length; j++) {
                const dx = this.particles[i].x - this.particles[j].x;
                const dy = this.particles[i].y - this.particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 120) {
                    this.ctx.strokeStyle = `rgba(79,142,247,${0.06 * (1 - dist / 120)})`;
                    this.ctx.lineWidth = 0.8;
                    this.ctx.beginPath();
                    this.ctx.moveTo(this.particles[i].x, this.particles[i].y);
                    this.ctx.lineTo(this.particles[j].x, this.particles[j].y);
                    this.ctx.stroke();
                }
            }
        }
    }

    animate() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.drawConnections();
        this.particles.forEach(p => {
            p.x += p.vx; p.y += p.vy;
            if (p.x < 0 || p.x > this.canvas.width)  p.vx *= -1;
            if (p.y < 0 || p.y > this.canvas.height) p.vy *= -1;
            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            this.ctx.fillStyle = `rgba(${p.color},${p.opacity})`;
            this.ctx.fill();
        });
        requestAnimationFrame(() => this.animate());
    }
}

// ─── Upload Zone Drag & Drop ──────────────────────────────
class UploadZone {
    constructor(el, onDrop) {
        this.el = el;
        this.onDrop = onDrop;
        this.bindEvents();
    }

    bindEvents() {
        ['dragenter','dragover','dragleave','drop'].forEach(evt => {
            this.el.addEventListener(evt, e => {
                e.preventDefault(); e.stopPropagation();
            });
        });
        this.el.addEventListener('dragenter', () => this.highlight(true));
        this.el.addEventListener('dragover',  () => this.highlight(true));
        this.el.addEventListener('dragleave', () => this.highlight(false));
        this.el.addEventListener('drop', e => {
            this.highlight(false);
            const files = Array.from(e.dataTransfer.files);
            if (this.onDrop) this.onDrop(files);
        });
    }

    highlight(active) {
        this.el.style.borderColor = active ? '#4f8ef7' : 'rgba(79,142,247,0.35)';
        this.el.style.background  = active ? 'rgba(79,142,247,0.06)' : '';
        this.el.style.transform   = active ? 'scale(1.01)' : '';
    }
}

// ─── Score Gauge Animation ────────────────────────────────
function animateScore(canvasId, score, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const cx = canvas.width / 2, cy = canvas.height / 2;
    const r = Math.min(cx, cy) - 12;
    let current = 0;
    const target = score / 100;

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        // Track
        ctx.beginPath();
        ctx.arc(cx, cy, r, -Math.PI / 2, Math.PI * 1.5);
        ctx.strokeStyle = 'rgba(255,255,255,0.06)';
        ctx.lineWidth = 10; ctx.lineCap = 'round';
        ctx.stroke();
        // Fill
        ctx.beginPath();
        ctx.arc(cx, cy, r, -Math.PI / 2, -Math.PI / 2 + current * Math.PI * 2);
        ctx.strokeStyle = color;
        ctx.lineWidth = 10; ctx.lineCap = 'round';
        ctx.stroke();
        // Text
        ctx.fillStyle = '#f0f4ff';
        ctx.font = 'bold 22px Inter';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(Math.round(current * 100) + '%', cx, cy);

        if (current < target) {
            current = Math.min(current + 0.012, target);
            requestAnimationFrame(draw);
        }
    }
    draw();
}

// ─── Counter Animation ────────────────────────────────────
function animateCounter(el, target, suffix = '') {
    let current = 0;
    const step = target / 60;
    const interval = setInterval(() => {
        current = Math.min(current + step, target);
        el.textContent = Math.round(current) + suffix;
        if (current >= target) clearInterval(interval);
    }, 16);
}

// ─── Progress Bar Animation ───────────────────────────────
function animateProgressBars() {
    document.querySelectorAll('.progress-fill').forEach(bar => {
        const width = bar.getAttribute('data-width') || '0';
        bar.style.setProperty('--width', width + '%');
        bar.style.width = width + '%';
        bar.style.transition = 'width 1.2s cubic-bezier(0.16, 1, 0.3, 1)';
    });
}

// ─── Intersection Observer for Fade-in ───────────────────
function initScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1 });
    document.querySelectorAll('.animate-on-scroll').forEach(el => observer.observe(el));
}

// ─── Toast Notifications ─────────────────────────────────
function showToast(msg, type = 'info') {
    const colors = { info: '#4f8ef7', success: '#10b981', warning: '#f59e0b', error: '#ef4444' };
    const icons  = { info: 'ℹ️', success: '✅', warning: '⚠️', error: '❌' };
    const toast  = document.createElement('div');
    toast.style.cssText = `
        position: fixed; bottom: 24px; right: 24px; z-index: 9999;
        background: rgba(15,22,40,0.96); backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.08);
        border-left: 4px solid ${colors[type]};
        border-radius: 12px; padding: 14px 20px;
        font-family: Inter, sans-serif; font-size: 0.9rem; color: #f0f4ff;
        display: flex; align-items: center; gap: 10px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        animation: slideUp 0.3s ease-out;
        max-width: 360px;
    `;
    toast.innerHTML = `<span>${icons[type]}</span><span>${msg}</span>`;
    document.body.appendChild(toast);
    setTimeout(() => { toast.style.opacity = '0'; toast.style.transition = 'opacity 0.3s'; setTimeout(() => toast.remove(), 300); }, 3500);
}

// ─── Skill Chart (Canvas) ────────────────────────────────
function drawSkillChart(canvasId, skills) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !skills.length) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const colors = ['#4f8ef7','#8b5cf6','#06b6d4','#10b981','#f59e0b','#ef4444'];
    const barH = 32, gap = 14;
    const maxVal = Math.max(...skills.map(s => s.count));
    const labelW = 120;

    canvas.height = skills.length * (barH + gap) + 40;
    ctx.clearRect(0, 0, W, canvas.height);

    skills.forEach((skill, i) => {
        const y = 20 + i * (barH + gap);
        const barW = ((W - labelW - 60) * skill.count) / maxVal;

        ctx.fillStyle = '#94a3b8';
        ctx.font = '13px Inter';
        ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
        ctx.fillText(skill.name, labelW - 10, y + barH / 2);

        const grad = ctx.createLinearGradient(labelW, y, labelW + barW, y);
        grad.addColorStop(0, colors[i % colors.length]);
        grad.addColorStop(1, colors[(i + 1) % colors.length]);

        ctx.fillStyle = 'rgba(255,255,255,0.04)';
        ctx.beginPath();
        ctx.roundRect(labelW, y, W - labelW - 60, barH, 6);
        ctx.fill();

        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.roundRect(labelW, y, barW, barH, 6);
        ctx.fill();

        ctx.fillStyle = '#f0f4ff';
        ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
        ctx.fillText(skill.count, labelW + barW + 8, y + barH / 2);
    });
}

// ─── Copy to Clipboard ───────────────────────────────────
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => showToast('Copied!', 'success'));
}

// ─── Theme Toggle ────────────────────────────────────────
function toggleTheme() {
    document.documentElement.classList.toggle('light-mode');
    const isDark = !document.documentElement.classList.contains('light-mode');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
}

// ─── Init ────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    // Particle canvas
    const canvas = document.getElementById('particle-canvas');
    if (canvas) new ParticleSystem(canvas);

    // Scroll animations
    initScrollAnimations();

    // Progress bars
    animateProgressBars();

    // Score gauges
    document.querySelectorAll('[data-score]').forEach(el => {
        const id = el.id, score = parseInt(el.getAttribute('data-score'));
        const color = score >= 80 ? '#10b981' : score >= 60 ? '#4f8ef7' : score >= 40 ? '#f59e0b' : '#ef4444';
        animateScore(id, score, color);
    });

    // Counters
    document.querySelectorAll('[data-count]').forEach(el => {
        const val = parseInt(el.getAttribute('data-count'));
        const suffix = el.getAttribute('data-suffix') || '';
        animateCounter(el, val, suffix);
    });

    // Upload zones
    document.querySelectorAll('.upload-zone').forEach(zone => {
        new UploadZone(zone, (files) => {
            showToast(`${files.length} file(s) ready to upload!`, 'success');
        });
    });

    showToast('Welcome to AI Resume Screener! 🚀', 'info');
});
