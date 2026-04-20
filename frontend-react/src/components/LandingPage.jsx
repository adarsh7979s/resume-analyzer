import React, { useState, useEffect, useRef } from 'react';
import { motion, useInView } from 'framer-motion';
import { Upload, FileText, ArrowRight, ChevronRight, TrendingUp, Shield, Sparkles, CheckCircle, Users, BarChart3 } from 'lucide-react';
import './LandingPage.css';

/* ── Typewriter hook ── */
function useTypewriter(words, typingMs = 100, pauseMs = 2200) {
  const [text, setText] = useState('');
  const [wordIdx, setWordIdx] = useState(0);
  const [isDeleting, setIsDeleting] = useState(false);

  useEffect(() => {
    const word = words[wordIdx];
    const timeout = isDeleting ? typingMs / 2 : typingMs;

    const timer = setTimeout(() => {
      if (!isDeleting) {
        setText(word.slice(0, text.length + 1));
        if (text.length + 1 === word.length) {
          setTimeout(() => setIsDeleting(true), pauseMs);
        }
      } else {
        setText(word.slice(0, text.length - 1));
        if (text.length === 0) {
          setIsDeleting(false);
          setWordIdx((prev) => (prev + 1) % words.length);
        }
      }
    }, timeout);
    return () => clearTimeout(timer);
  }, [text, isDeleting, wordIdx, words, typingMs, pauseMs]);

  return text;
}

/* ── Animated counter ── */
function AnimCounter({ value, suffix = '', duration = 1400 }) {
  const [display, setDisplay] = useState(0);
  const ref = useRef(null);
  const inView = useInView(ref, { once: true });

  useEffect(() => {
    if (!inView) return;
    const t0 = performance.now();
    function tick(now) {
      const p = Math.min((now - t0) / duration, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      setDisplay(Math.round(eased * value));
      if (p < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }, [inView, value, duration]);

  return <span ref={ref}>{display.toLocaleString()}{suffix}</span>;
}

/* ── ScrollReveal wrapper ── */
function Reveal({ children, delay = 0, className = '' }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: '-80px' });
  return (
    <motion.div
      ref={ref}
      className={className}
      initial={{ opacity: 0, y: 32 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.7, delay, ease: [0.16, 1, 0.3, 1] }}
    >
      {children}
    </motion.div>
  );
}

/* ── Mouse glow card ── */
function GlowCard({ children, className = '', style = {} }) {
  const ref = useRef(null);
  const [pos, setPos] = useState({ x: 0, y: 0 });
  const [hovering, setHovering] = useState(false);

  function handleMove(e) {
    const rect = ref.current?.getBoundingClientRect();
    if (!rect) return;
    setPos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
  }

  return (
    <div
      ref={ref}
      className={`glow-card ${className}`}
      style={style}
      onMouseMove={handleMove}
      onMouseEnter={() => setHovering(true)}
      onMouseLeave={() => setHovering(false)}
    >
      {hovering && (
        <div
          className="glow-card-glow"
          style={{ left: pos.x, top: pos.y }}
        />
      )}
      <div className="glow-card-inner">{children}</div>
    </div>
  );
}

export default function LandingPage({ onStart }) {
  const typedWord = useTypewriter(['Trajectory.', 'Potential.', 'Future.', 'Career.']);

  const features = [
    { color: 'var(--accent)', icon: <Shield size={22} />, title: 'ATS-Ready Analysis', desc: 'Scan your resume against 30+ ATS criteria and get instant optimization tips to beat applicant tracking systems.' },
    { color: 'var(--info)',   icon: <BarChart3 size={22} />, title: 'Skill Benchmarking', desc: 'Map your capabilities against thousands of live job postings to see exactly where you stand versus the competition.' },
    { color: 'var(--warning)',icon: <TrendingUp size={22} />, title: 'Growth Roadmap', desc: 'AI-generated learning paths with curated courses to close your most impactful skill gaps in weeks, not months.' },
  ];

  const stats = [
    { value: 30000, suffix: '+', label: 'Resumes Analyzed' },
    { value: 87, suffix: '%', label: 'Avg. Score Improvement' },
    { value: 150, suffix: '+', label: 'Job Roles Covered' },
    { value: 42, suffix: '%', label: 'Higher Response Rate' },
  ];

  return (
    <div className="landing">
      {/* Navbar */}
      <nav className="noir-nav">
        <div className="noir-nav-brand">
          <span className="brand-diamond" />
          Career Copilot
        </div>
        <div className="noir-nav-links">
          <a href="#features">Features</a>
          <a href="#stats">Results</a>
        </div>
        <button className="nav-access-btn" onClick={onStart}>
          Get Started <ChevronRight size={11} />
        </button>
      </nav>

      <main className="landing-main">
        {/* ── HERO ── */}
        <section className="landing-hero">
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
          >
            {/* Badge */}
            <div className="hero-badge">
              <span className="badge-ping">
                <span className="badge-ping-bg" />
                <span className="badge-dot-inner" />
              </span>
              AI-Powered Resume Intelligence — v2.0 live
              <ChevronRight size={11} style={{ color: 'var(--accent)' }} />
            </div>
          </motion.div>

          <motion.h1
            className="landing-title"
            initial={{ opacity: 0, y: 32 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.9, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
          >
            Land Your Dream Role.<br />
            Master Your{' '}
            <span className="title-accent">
              {typedWord}
              <span className="type-cursor">|</span>
            </span>
          </motion.h1>

          <motion.p
            className="landing-subtitle"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.25 }}
          >
            Upload your resume, set your target role, and get a precision skill-gap audit with
            personalized strategies — all powered by advanced AI in under 60 seconds.
          </motion.p>

          <motion.div
            className="landing-cta-row"
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.4 }}
          >
            <button className="btn-shiny" onClick={onStart}>
              <Sparkles size={16} /> Start Free Analysis <ArrowRight size={16} />
            </button>
            <button className="btn-ghost" onClick={onStart}>
              <FileText size={15} /> See how it works
            </button>
          </motion.div>

          {/* Trust strip */}
          <motion.div
            className="trust-strip"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.6 }}
          >
            <p className="trust-label">Powered by</p>
            <div className="trust-logos">
              {['OpenAI', 'FastAPI', 'React', 'Vercel'].map((n) => (
                <div key={n} className="trust-item">
                  <span className="trust-dot" />
                  {n}
                </div>
              ))}
            </div>
          </motion.div>
        </section>

        {/* ── STAT COUNTERS ── */}
        <section className="stats-strip" id="stats">
          {stats.map((s, i) => (
            <Reveal key={s.label} delay={i * 0.08} className="stat-cell">
              <strong className="stat-value">
                <AnimCounter value={s.value} suffix={s.suffix} />
              </strong>
              <span className="stat-label">{s.label}</span>
            </Reveal>
          ))}
        </section>

        {/* ── FEATURE BENTO ── */}
        <section className="feature-section" id="features">
          <Reveal>
            <div className="feature-section-head">
              <h2>
                The Operating System for<br />
                <span style={{ color: 'var(--accent)' }}>Your Career Strategy</span>
              </h2>
              <p>Replace scattered guesswork with one AI-driven intelligence platform.</p>
            </div>
          </Reveal>

          <div className="feature-bento">
            {/* Main large card */}
            <Reveal delay={0.1}>
              <GlowCard className="feature-card feature-card-main">
                <div className="feature-icon-box" style={{ color: 'var(--accent)' }}>
                  <Upload size={22} />
                </div>
                <h3>Deep Gap Analysis</h3>
                <p>
                  Our AI performs a comprehensive architectural scan of your professional narrative to match
                  your trajectory against thousands of real-world role requirements with precision scoring.
                </p>
                <div className="feature-card-footer">
                  <span>EXPLORE FEATURE</span>
                  <ArrowRight size={13} />
                </div>
              </GlowCard>
            </Reveal>

            {features.slice(1).map((f, i) => (
              <Reveal key={f.title} delay={0.15 + i * 0.08}>
                <GlowCard className="feature-card">
                  <div className="feature-icon-box" style={{ color: f.color }}>
                    {f.icon}
                  </div>
                  <h3>{f.title}</h3>
                  <p>{f.desc}</p>
                </GlowCard>
              </Reveal>
            ))}
          </div>
        </section>

        {/* ── SOCIAL PROOF (no name) ── */}
        <section className="proof-section">
          <Reveal className="proof-card">
            <div className="proof-stars">
              {[...Array(5)].map((_, i) => (
                <svg key={i} width="20" height="20" viewBox="0 0 20 20" fill="var(--accent)">
                  <path d="M10 15l-5.878 3.09 1.123-6.545L.489 6.91l6.572-.955L10 0l2.939 5.955 6.572.955-4.756 4.635 1.123 6.545z" />
                </svg>
              ))}
            </div>
            <blockquote>
              "This tool completely transformed how I approach job applications. Within a week I went
              from a 40% to an 87% match score and landed three interviews."
            </blockquote>
            <div className="proof-chips">
              <span className="proof-chip"><CheckCircle size={12} /> 30% higher chance of getting a job</span>
              <span className="proof-chip"><CheckCircle size={12} /> 42% higher response rate from recruiters</span>
            </div>
          </Reveal>
        </section>

        {/* ── FINAL CTA ── */}
        <section className="final-cta">
          <Reveal>
            <h2>
              Ready to <span style={{ color: 'var(--accent)' }}>Level Up?</span>
            </h2>
          </Reveal>
          <Reveal delay={0.1}>
            <p>Start your career intelligence audit in under 60 seconds. Completely free.</p>
          </Reveal>
          <Reveal delay={0.2}>
            <button className="btn-shiny" onClick={onStart}>
              <Sparkles size={16} /> Start Free Analysis <ArrowRight size={16} />
            </button>
          </Reveal>
        </section>
      </main>

      {/* Footer */}
      <footer className="landing-footer">
        <div className="footer-inner">
          <div className="footer-brand">
            <span className="brand-diamond" />
            <span>Career Copilot</span>
          </div>
          <p className="footer-copy">&copy; 2024 Career Copilot. Built with AI for ambitious professionals.</p>
        </div>
      </footer>
    </div>
  );
}
