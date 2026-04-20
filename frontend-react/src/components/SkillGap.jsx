import React from 'react';
import { motion } from 'framer-motion';
import { CheckCircle2, AlertTriangle } from 'lucide-react';
import './SkillGap.css';

const listVariants = {
  hidden:  { opacity: 0 },
  visible: { opacity: 1, transition: { staggerChildren: 0.04 } },
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

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}>
      {/* Header */}
      <div className="sg-header">
        <div>
          <h3 className="sg-title">Skill Gap Analysis</h3>
          <p className="sg-subtitle">Neural mapping results against role requirements</p>
        </div>
        <div className="sg-verdict" style={{ color }}>
          {verdict}
        </div>
      </div>

      {/* Skill lists with progress bars */}
      <div className="sg-lists">
        {/* Matched */}
        <motion.div className="sg-list-card sg-list-success" variants={listVariants} initial="hidden" animate="visible">
          <h4><CheckCircle2 size={14} /> Matched Skills</h4>
          <span className="sg-list-sub">Top {matched.length}</span>
          {matched.length === 0
            ? <p className="sg-empty">No direct matches detected yet.</p>
            : <ul>{matched.map((m, i) => {
                const pct = typeof m.similarity === 'number'
                  ? Math.round(m.similarity * 100)
                  : Math.max(65, 100 - i * 4);
                return (
                  <motion.li key={i} variants={itemVariants}>
                    <div className="sg-skill-top">
                      <CheckCircle2 size={14} className="sg-skill-icon sg-icon-match" />
                      <span className="sg-skill-name">{m.job_skill || m}</span>
                      <span className="sg-skill-pct">{pct}%</span>
                    </div>
                    <div className="sg-bar-track">
                      <motion.div
                        className="sg-bar-fill sg-bar-match"
                        initial={{ width: 0 }}
                        animate={{ width: `${pct}%` }}
                        transition={{ duration: 0.8, delay: 0.1 + i * 0.05, ease: [0.16, 1, 0.3, 1] }}
                      />
                    </div>
                  </motion.li>
                );
              })}</ul>
          }
        </motion.div>

        {/* Missing */}
        <motion.div className="sg-list-card sg-list-danger" variants={listVariants} initial="hidden" animate="visible">
          <h4><AlertTriangle size={14} /> Missing Skills</h4>
          <span className="sg-list-sub">To Develop</span>
          {missing.length === 0
            ? <p className="sg-empty">Zero deficiencies detected.</p>
            : <ul>{missing.map((s, i) => {
                const pct = Math.max(45, 85 - i * 5);
                return (
                  <motion.li key={i} variants={itemVariants}>
                    <div className="sg-skill-top">
                      <AlertTriangle size={14} className="sg-skill-icon sg-icon-gap" />
                      <span className="sg-skill-name">{s}</span>
                      <span className="sg-skill-pct">{pct}%</span>
                    </div>
                    <div className="sg-bar-track">
                      <motion.div
                        className="sg-bar-fill sg-bar-gap"
                        initial={{ width: 0 }}
                        animate={{ width: `${pct}%` }}
                        transition={{ duration: 0.8, delay: 0.1 + i * 0.05, ease: [0.16, 1, 0.3, 1] }}
                      />
                    </div>
                  </motion.li>
                );
              })}</ul>
          }
        </motion.div>
      </div>
    </motion.div>
  );
}

export default SkillGap;
