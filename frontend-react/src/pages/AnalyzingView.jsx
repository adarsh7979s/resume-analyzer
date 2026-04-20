import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import { Check, XCircle } from 'lucide-react';

const STEPS = [
  'Mapping role requirements',
  'Matching resume against role',
  'Generating AI recommendations',
];

export default function AnalyzingView({ state }) {
  const { analysisProgress, startNewResumeFlow } = state;
  const pct = useMemo(() => Math.min(95, analysisProgress * 32), [analysisProgress]);

  return (
    <motion.div
      className="stage"
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 1.02 }}
      transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
    >
      <div className="stage-card" style={{ textAlign: 'center' }}>

        {/* Header row */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 40 }}>
          <div className="kicker">
            <span className="kicker-dot" />
            Analysis In Progress
          </div>
          <button
            className="btn-ghost"
            style={{ padding: '6px 16px', fontSize: '0.75rem' }}
            onClick={startNewResumeFlow}
          >
            <XCircle size={14} /> Cancel
          </button>
        </div>

        {/* ── Orbital Scanner ── */}
        <div className="scanner-wrap">
          <motion.div
            className="scanner-ring scanner-ring-outer"
            animate={{ scale: [1, 1.5], opacity: [0.3, 0] }}
            transition={{ repeat: Infinity, duration: 2 }}
          />
          <motion.div
            className="scanner-ring scanner-ring-mid"
            animate={{ scale: [1, 1.3], opacity: [0.2, 0] }}
            transition={{ repeat: Infinity, duration: 2, delay: 0.5 }}
          />
          <motion.div
            style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
            animate={{ rotate: 360 }}
            transition={{ repeat: Infinity, duration: 3.5, ease: 'linear' }}
          >
            <div className="scanner-orbit-dot" />
          </motion.div>
          <div className="scanner-center">
            <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
              <circle cx="20" cy="20" r="18" stroke="var(--accent)" strokeWidth="1.5" strokeDasharray="4 4" />
              <circle cx="20" cy="20" r="8" fill="var(--accent)" opacity="0.15" />
              <circle cx="20" cy="20" r="3" fill="var(--accent)" />
            </svg>
          </div>
        </div>

        {/* Headline */}
        <h1 className="stage-title" style={{ marginBottom: 12, marginTop: 40 }}>
          Analyzing Your Profile.
        </h1>
        <p className="stage-sub" style={{ maxWidth: 440, margin: '0 auto 40px' }}>
          Our AI engine is mapping your professional skills against real-world
          role requirements and generating recommendations.
        </p>

        {/* Progress bar */}
        <div className="progress-track" style={{ marginBottom: 40 }}>
          <motion.div
            className="progress-fill"
            initial={{ width: 0 }}
            animate={{ width: `${pct}%` }}
            transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
          />
        </div>

        {/* Steps */}
        <div style={{ textAlign: 'left', maxWidth: 320, margin: '0 auto' }}>
          {STEPS.map((step, idx) => {
            const order = idx + 1;
            const done   = analysisProgress > order;
            const active = analysisProgress === order;
            return (
              <motion.div
                key={step}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.12 }}
                style={{
                  display: 'flex', alignItems: 'center', gap: 14,
                  padding: '12px 0',
                  opacity: done || active ? 1 : 0.25,
                }}
              >
                <div style={{
                  width: 24, height: 24,
                  borderRadius: 6,
                  background: done ? 'var(--success)' : active ? 'var(--accent)' : 'rgba(255,255,255,0.08)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  flexShrink: 0,
                  transition: 'background 0.3s',
                }}>
                  {done
                    ? <Check size={14} color="#000" strokeWidth={3} />
                    : active
                      ? <motion.span
                          style={{ width: 8, height: 8, background: '#fff', borderRadius: '50%', display: 'block' }}
                          animate={{ opacity: [1, 0.3, 1] }}
                          transition={{ repeat: Infinity, duration: 1.2 }}
                        />
                      : null}
                </div>
                <span style={{
                  fontSize: '0.88rem',
                  color: done ? 'var(--success)' : active ? 'var(--txt)' : 'var(--txt-3)',
                  fontWeight: active ? 700 : 500,
                }}>
                  {step}
                </span>
              </motion.div>
            );
          })}
        </div>
      </div>

      <style>{`
        .scanner-wrap {
          position: relative;
          width: 120px; height: 120px;
          margin: 0 auto;
        }
        .scanner-ring {
          position: absolute;
          border-radius: 50%;
          border: 2px solid var(--accent);
        }
        .scanner-ring-outer { inset: 0; }
        .scanner-ring-mid   { inset: 14px; border-color: rgba(20,184,166,0.4); }
        .scanner-orbit-dot {
          position: absolute;
          top: 0; left: 50%;
          width: 10px; height: 10px;
          background: var(--accent);
          border-radius: 50%;
          box-shadow: 0 0 16px var(--accent-glow);
          margin-top: -5px; margin-left: -5px;
        }
        .scanner-center {
          position: absolute; inset: 0;
          display: flex; align-items: center; justify-content: center;
        }
        .progress-track {
          height: 6px;
          background: rgba(255,255,255,0.07);
          border-radius: 99px; overflow: hidden;
          max-width: 360px; margin: 0 auto;
        }
        .progress-fill {
          height: 100%;
          background: linear-gradient(90deg, var(--accent), var(--info));
          border-radius: 99px;
          box-shadow: 0 0 12px var(--accent-glow);
        }
      `}</style>
    </motion.div>
  );
}
