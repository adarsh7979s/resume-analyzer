import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import './SkillGap.css';

const listVariants = {
  hidden:  { opacity: 0 },
  visible: { opacity: 1, transition: { staggerChildren: 0.05 } },
};
const itemVariants = {
  hidden:  { opacity: 0, y: 6 },
  visible: { opacity: 1, y: 0, transition: { type: 'spring', stiffness: 300, damping: 24 } },
};

function SkillGap({ score, matched, missing }) {
  let verdict, color;
  if      (score >= 80) { verdict = 'Elite Calibration'; color = 'var(--success)'; }
  else if (score >= 60) { verdict = 'Strong Alignment';  color = 'var(--accent)'; }
  else if (score >= 40) { verdict = 'Moderate Offset';   color = 'var(--warning)'; }
  else                  { verdict = 'Critical Gap';      color = 'var(--danger)'; }

  const [displayScore, setDisplayScore] = useState(0);
  useEffect(() => {
    const t0 = performance.now();
    function tick(now) {
      const p = Math.min((now - t0) / 1000, 1);
      setDisplayScore(Math.round(p * score));
      if (p < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }, [score]);

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 24 }}>
        <div>
          <h3 style={{ fontFamily: 'var(--font-head)', fontWeight: 800, fontSize: '1.1rem', marginBottom: 4 }}>
            Capability Benchmark
          </h3>
          <p style={{ fontSize: '0.78rem', color: 'var(--txt-3)' }}>Neural mapping results against global standards</p>
        </div>
        <div style={{ textAlign: 'right' }}>
          <span style={{ fontFamily: 'var(--font-head)', fontWeight: 800, fontSize: '0.95rem', color }}>{verdict}</span>
          <p style={{ fontSize: '0.6rem', color: 'var(--txt-3)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', marginTop: 2 }}>Verdict</p>
        </div>
      </div>

      {/* Gauge */}
      <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 28 }}>
        <div className="sg-circle-wrap">
          <div className="sg-circle" style={{ background: `conic-gradient(${color} ${displayScore * 3.6}deg, rgba(255,255,255,0.05) 0deg)` }} />
          <div className="sg-circle-inner">
            <motion.span initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
              {displayScore}<small>%</small>
            </motion.span>
            <p>MATCH INDEX</p>
          </div>
        </div>
      </div>

      {/* Skill lists */}
      <div className="sg-lists">
        <motion.div className="sg-list-card sg-list-success" variants={listVariants} initial="hidden" animate="visible">
          <h4>Identified Strengths</h4>
          {matched.length === 0
            ? <p className="sg-empty">No direct matches detected yet.</p>
            : <ul>{matched.map((m, i) => (
                <motion.li key={i} variants={itemVariants}>
                  <span>{m.job_skill}</span>
                  <span className="sg-tag sg-tag-match">MATCH</span>
                </motion.li>
              ))}</ul>
          }
        </motion.div>

        <motion.div className="sg-list-card sg-list-danger" variants={listVariants} initial="hidden" animate="visible">
          <h4>Critical Targets</h4>
          {missing.length === 0
            ? <p className="sg-empty">Zero deficiencies detected.</p>
            : <ul>{missing.map((s, i) => (
                <motion.li key={i} variants={itemVariants}>
                  <span>{s}</span>
                  <span className="sg-tag sg-tag-gap">UPSKILL</span>
                </motion.li>
              ))}</ul>
          }
        </motion.div>
      </div>
    </motion.div>
  );
}

export default SkillGap;
