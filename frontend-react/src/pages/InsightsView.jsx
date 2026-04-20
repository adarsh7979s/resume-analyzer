import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  TrendingUp, Target, Award, BookOpen, Lightbulb,
  Briefcase, GraduationCap, ExternalLink,
} from 'lucide-react';

const item = {
  hidden:  { opacity: 0, y: 14 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.45, ease: [0.16, 1, 0.3, 1] } },
};

export default function InsightsView({ state }) {
  const {
    score, atsScore, matched, missing,
    resumeSkills, recommendations,
    candidateName, role,
    insightTab, setInsightTab,
  } = state;

  const hasResults = score !== null;
  const tabs = [
    { id: 'strategy', label: 'Strategy', icon: <Target size={14} /> },
    { id: 'academy',  label: 'Academy',  icon: <GraduationCap size={14} /> },
    { id: 'resume',   label: 'Resume',   icon: <Briefcase size={14} /> },
  ];

  /* ── Strategy content ── */
  function renderStrategy() {
    if (!hasResults) return renderEmpty('Complete an analysis to see your career strategy.');

    const strengthPct = matched.length / Math.max(1, matched.length + missing.length) * 100;
    return (
      <div className="ins-section">
        <div className="ins-stat-grid">
          <div className="ins-stat-card">
            <div className="ins-stat-icon" style={{ background: 'rgba(74,222,128,0.1)', color: 'var(--success)' }}>
              <Award size={20} />
            </div>
            <div className="ins-stat-val">{matched.length}</div>
            <div className="ins-stat-label">Matched Skills</div>
          </div>
          <div className="ins-stat-card">
            <div className="ins-stat-icon" style={{ background: 'rgba(251,191,36,0.1)', color: 'var(--warning)' }}>
              <Lightbulb size={20} />
            </div>
            <div className="ins-stat-val">{missing.length}</div>
            <div className="ins-stat-label">Skills to Learn</div>
          </div>
          <div className="ins-stat-card">
            <div className="ins-stat-icon" style={{ background: 'rgba(20,184,166,0.1)', color: 'var(--accent)' }}>
              <TrendingUp size={20} />
            </div>
            <div className="ins-stat-val">{Math.round(strengthPct)}%</div>
            <div className="ins-stat-label">Coverage</div>
          </div>
        </div>

        {recommendations?.summary && (
          <div className="ins-summary-card">
            <h4><Lightbulb size={14} /> AI Summary</h4>
            <p>{recommendations.summary}</p>
          </div>
        )}

        {recommendations?.focus_areas?.length > 0 && (
          <div className="ins-list-card">
            <h4><Target size={14} /> Focus Areas</h4>
            <ul>
              {recommendations.focus_areas.map((area, i) => (
                <li key={i}>{area}</li>
              ))}
            </ul>
          </div>
        )}

        {recommendations?.action_plan?.length > 0 && (
          <div className="ins-list-card">
            <h4><Award size={14} /> Action Plan</h4>
            <ol>
              {recommendations.action_plan.map((step, i) => (
                <li key={i}>{step}</li>
              ))}
            </ol>
          </div>
        )}
      </div>
    );
  }

  /* ── Academy content ── */
  function renderAcademy() {
    if (!hasResults || !recommendations?.courses?.length) {
      return renderEmpty('Course suggestions appear after analysis.');
    }
    return (
      <div className="ins-section">
        <div className="ins-courses-grid">
          {recommendations.courses.map((c, i) => (
            <motion.div key={i} className="ins-course-card" variants={item} initial="hidden" animate="visible">
              <div className="ins-course-platform">{c.platform || 'Online'}</div>
              <h4>{c.title}</h4>
              <p>{c.description || `Strengthen your ${c.skill || 'skills'} with this course.`}</p>
              <div className="ins-course-footer">
                <span className="ins-course-skill">{c.skill || missing[i] || 'General'}</span>
                {c.url && (
                  <a href={c.url} target="_blank" rel="noopener noreferrer" className="ins-course-link">
                    <ExternalLink size={12} /> View
                  </a>
                )}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    );
  }

  /* ── Resume feedback content ── */
  function renderResume() {
    if (!hasResults) return renderEmpty('Upload and analyze a resume to see feedback.');

    return (
      <div className="ins-section">
        <div className="ins-resume-stats">
          <div className="ins-resume-stat">
            <span className="ins-resume-stat-label">Match Score</span>
            <span className="ins-resume-stat-val" style={{ color: score >= 70 ? 'var(--success)' : 'var(--accent)' }}>
              {score}%
            </span>
          </div>
          <div className="ins-resume-stat">
            <span className="ins-resume-stat-label">ATS Score</span>
            <span className="ins-resume-stat-val" style={{ color: 'var(--accent)' }}>
              {atsScore}%
            </span>
          </div>
          <div className="ins-resume-stat">
            <span className="ins-resume-stat-label">Skills Found</span>
            <span className="ins-resume-stat-val">{resumeSkills.length}</span>
          </div>
        </div>

        <div className="ins-list-card">
          <h4><Briefcase size={14} /> Extracted Skills</h4>
          <div className="ins-skill-tags">
            {resumeSkills.map((skill, i) => (
              <span key={i} className="ins-skill-tag">{skill}</span>
            ))}
          </div>
        </div>

        {matched.length > 0 && (
          <div className="ins-list-card">
            <h4><Award size={14} /> Top Strengths</h4>
            <ul>
              {matched.slice(0, 5).map((m, i) => (
                <li key={i}>
                  <strong>{m.job_skill || m}</strong>
                  {m.similarity && <span className="ins-sim"> ({Math.round(m.similarity * 100)}% match)</span>}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    );
  }

  function renderEmpty(message) {
    return (
      <div className="ins-empty">
        <BookOpen size={40} strokeWidth={1} />
        <p>{message}</p>
      </div>
    );
  }

  return (
    <div className="ins-wrap">
      <div className="ins-header">
        <div>
          <h2 className="ins-title">Career Insights</h2>
          <p className="ins-subtitle">
            {hasResults
              ? `${candidateName || 'Your'} analysis for ${role}`
              : 'AI-powered career strategy & recommendations'}
          </p>
        </div>
      </div>

      <div className="ins-tabs">
        {tabs.map(tab => (
          <button
            key={tab.id}
            className={`ins-tab ${insightTab === tab.id ? 'active' : ''}`}
            onClick={() => setInsightTab(tab.id)}
          >
            {tab.icon} {tab.label}
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={insightTab}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.3 }}
        >
          {insightTab === 'strategy' && renderStrategy()}
          {insightTab === 'academy'  && renderAcademy()}
          {insightTab === 'resume'   && renderResume()}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
