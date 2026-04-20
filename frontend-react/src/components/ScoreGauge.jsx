import React, { useEffect, useState, useRef } from 'react';
import { useInView } from 'framer-motion';
import './ScoreGauge.css';

const CIRCUMFERENCE = 2 * Math.PI * 65; // radius=65

export default function ScoreGauge({ value, label, color, metrics }) {
  const [animatedValue, setAnimatedValue] = useState(0);
  const ref = useRef(null);
  const inView = useInView(ref, { once: true });

  useEffect(() => {
    if (!inView || typeof value !== 'number') return;
    const t0 = performance.now();
    function tick(now) {
      const p = Math.min((now - t0) / 1200, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      setAnimatedValue(Math.round(eased * value));
      if (p < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }, [inView, value]);

  const dashOffset = CIRCUMFERENCE - (CIRCUMFERENCE * (animatedValue / 100));

  const gaugeColor = color
    || (value >= 80 ? 'var(--success)' : value >= 50 ? 'var(--accent)' : 'var(--danger)');

  return (
    <div className="gauge-wrap" ref={ref}>
      <div className="gauge-svg-wrap">
        <svg className="gauge-svg" viewBox="0 0 150 150">
          <defs>
            <filter id={`gauge-glow-${label}`} x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="4" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>
          <circle
            className="gauge-track"
            cx="75"
            cy="75"
            r="65"
          />
          <circle
            className="gauge-fill"
            cx="75"
            cy="75"
            r="65"
            stroke={gaugeColor}
            strokeDasharray={CIRCUMFERENCE}
            strokeDashoffset={dashOffset}
            style={{ filter: `url(#gauge-glow-${label})` }}
          />
        </svg>

        <div className="gauge-center">
          <span className="gauge-value">
            {typeof value === 'number' ? animatedValue : '–'}
            <small>%</small>
          </span>
        </div>
      </div>

      <span className="gauge-title">{label}</span>

      {metrics && metrics.length > 0 && (
        <div className="gauge-metrics">
          {metrics.map((m, i) => (
            <div key={i} className="gauge-metric-row">
              <span className="gauge-metric-dot" style={{ background: gaugeColor }} />
              <span className="gauge-metric-name">{m.name}</span>
              <span className="gauge-metric-value">{m.value}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
