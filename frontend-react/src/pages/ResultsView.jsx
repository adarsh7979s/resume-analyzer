import React, { useEffect, useState, useRef } from 'react';
import { motion, useInView } from 'framer-motion';
import { BarChart3, Target, AlertTriangle, Award, ArrowRight, RefreshCw } from 'lucide-react';
import toast from 'react-hot-toast';
import SkillGap from '../components/SkillGap';

/* ── Animated counter ── */
function AnimVal({ value, suffix = '' }) {
  const [display, setDisplay] = useState(0);
  const ref = useRef(null);
  const inView = useInView(ref, { once: true });

  useEffect(() => {
    if (!inView || typeof value !== 'number') return;
    const t0 = performance.now();
    function tick(now) {
      const p = Math.min((now - t0) / 900, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      setDisplay(Math.round(eased * value));
      if (p < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }, [inView, value]);

  return <span ref={ref}>{typeof value === 'number' ? display : value}{suffix}</span>;
}

export default function ResultsView({ state }) {
  const {
    startNewResumeFlow,
    resumeSkills, jobSkills,
    atsScore, score, matched, missing,
    status,
  } = state;

  const statusIsError = /failed|error|missing|expired|not found|unsupported/i.test(status || '');

  // Notify on load
  useEffect(() => {
    if (score !== null) {
      const msg = score >= 80
        ? `🎉 Excellent! ${score}% match — you're highly competitive.`
        : score >= 50
          ? `📊 ${score}% match — good foundation, room to grow.`
          : `⚡ ${score}% match — let's build a plan to improve.`;
      toast(msg, { duration: 4000 });
    }
  }, [score]);

  const kpis = [
    { label: 'Inventory', value: resumeSkills.length, sub: 'Extracted Skills', icon: <BarChart3 size={18} />, color: 'var(--accent)' },
    { label: 'Benchmark', value: jobSkills.length,    sub: 'Required Skills',  icon: <Target size={18} />,    color: 'var(--info)' },
    { label: 'ATS Index', value: atsScore === null ? '–' : atsScore, suffix: atsScore !== null ? '%' : '', sub: 'Search Visibility', icon: <Award size={18} />, color: 'var(--warning)' },
    { label: 'Skill Gap', value: score === null ? '–' : missing.length, sub: 'Critical Targets', icon: <AlertTriangle size={18} />, color: 'var(--danger)' },
  ];

  const container = {
    hidden:  { opacity: 0 },
    visible: { opacity: 1, transition: { staggerChildren: 0.08 } },
  };
  const item = {
    hidden:  { opacity: 0, y: 18 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.5, ease: [0.16, 1, 0.3, 1] } },
  };

  return (
    <motion.div
      className="results-main"
      variants={container}
      initial="hidden"
      animate="visible"
      exit={{ opacity: 0 }}
    >
      {/* Header */}
      <motion.header variants={item} className="results-header">
        <div>
          <div className="kicker">
            <span className="kicker-dot" />
            Analysis Complete
          </div>
          <h1 className="results-title">
            Your <span>Results.</span>
          </h1>
        </div>
        <button
          className="btn-ghost"
          style={{ padding: '10px 22px', fontSize: '0.82rem' }}
          onClick={startNewResumeFlow}
        >
          <RefreshCw size={14} /> New Analysis
        </button>
      </motion.header>

      {/* Score hero banner */}
      {score !== null && (
        <motion.div variants={item} className="score-hero">
          <div className="score-hero-left">
            <span className="score-hero-label">Match Score</span>
            <div className="score-hero-value" style={{
              color: score >= 80 ? 'var(--success)' : score >= 50 ? 'var(--accent)' : 'var(--danger)'
            }}>
              <AnimVal value={score} suffix="%" />
            </div>
            <span className="score-hero-verdict">
              {score >= 80 ? 'Elite Calibration' : score >= 60 ? 'Strong Alignment' : score >= 40 ? 'Moderate Fit' : 'Needs Work'}
            </span>
          </div>
          <div className="score-hero-bar-wrap">
            <motion.div
              className="score-hero-bar"
              initial={{ width: 0 }}
              animate={{ width: `${score}%` }}
              transition={{ duration: 1.2, delay: 0.3, ease: [0.16, 1, 0.3, 1] }}
              style={{
                background: score >= 80
                  ? 'linear-gradient(90deg, var(--success), var(--accent))'
                  : score >= 50
                    ? 'linear-gradient(90deg, var(--accent), var(--info))'
                    : 'linear-gradient(90deg, var(--danger), var(--warning))',
              }}
            />
          </div>
        </motion.div>
      )}

      {/* KPI Row */}
      <motion.div variants={item} className="kpi-row">
        {kpis.map((k) => (
          <motion.div
            key={k.label}
            className="kpi-card"
            whileHover={{ y: -3, borderColor: 'var(--border-teal)' }}
            transition={{ type: 'spring', stiffness: 400, damping: 20 }}
          >
            <div className="kpi-card-icon" style={{ color: k.color }}>{k.icon}</div>
            <span className="kpi-card-label">{k.label}</span>
            <strong className="kpi-card-value">
              <AnimVal value={k.value} suffix={k.suffix || ''} />
            </strong>
            <span className="kpi-card-sub">{k.sub}</span>
          </motion.div>
        ))}
      </motion.div>

      {/* Main analysis (SkillGap component) */}
      <motion.div variants={item} className="analysis-card">
        <SkillGap score={score || 0} matched={matched} missing={missing} />
      </motion.div>

      {/* Skills inventory */}
      <motion.div variants={item} className="skills-card">
        <h3>Expertise Inventory</h3>
        <details className="skills-disclosure" open>
          <summary>Detected Expertise ({resumeSkills.length})</summary>
          <ul className="skill-list">
            {resumeSkills.map((s, i) => <li key={`r-${i}`}>{s}</li>)}
          </ul>
        </details>
        <details className="skills-disclosure">
          <summary>Benchmark Requirements ({jobSkills.length})</summary>
          <ul className="skill-list">
            {jobSkills.map((s, i) => <li key={`j-${i}`} style={{ opacity: 0.6 }}>{s}</li>)}
          </ul>
        </details>
      </motion.div>

      {/* Status */}
      {status && (
        <motion.div variants={item}>
          <div className={`status-bar ${statusIsError ? 'status-error' : 'status-info'}`}>
            {status}
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}
