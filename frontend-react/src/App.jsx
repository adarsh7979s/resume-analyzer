import React, { useEffect, useState, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Sparkles, Layout, BookOpen, FileEdit, TrendingUp, Zap, ArrowRight } from 'lucide-react';
import { Toaster } from 'react-hot-toast';
import confetti from 'canvas-confetti';
import './App.css';
import { useAnalyzer } from './hooks/useAnalyzer';
import LandingPage from './components/LandingPage';
import RobotCompanion from './components/RobotCompanion';
import UploadView from './pages/UploadView';
import AnalyzingView from './pages/AnalyzingView';
import ResultsView from './pages/ResultsView';

/* ── Confetti burst on score ── */
function useConfetti(score) {
  useEffect(() => {
    if (typeof score !== 'number' || score < 70) return;
    const end = Date.now() + 1800;
    const colors = ['#14b8a6', '#6366f1', '#fbbf24', '#4ade80'];
    (function frame() {
      confetti({
        particleCount: 2,
        angle: 60,
        spread: 55,
        origin: { x: 0 },
        colors,
      });
      confetti({
        particleCount: 2,
        angle: 120,
        spread: 55,
        origin: { x: 1 },
        colors,
      });
      if (Date.now() < end) requestAnimationFrame(frame);
    })();
  }, [score]);
}

function App() {
  const state = useAnalyzer();
  const { view, setView, isAnalysisMode, isLoading } = state;

  // Fire confetti when results come in with a good score
  useConfetti(view === 'results' ? state.score : null);

  return (
    <div className="page">
      {/* Toast system */}
      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: 'rgba(10, 22, 40, 0.9)',
            color: '#f0f6ff',
            border: '1px solid rgba(20,184,166,0.2)',
            backdropFilter: 'blur(12px)',
            borderRadius: '12px',
            fontSize: '0.85rem',
          },
          success: { iconTheme: { primary: '#14b8a6', secondary: '#000' } },
          error:   { iconTheme: { primary: '#f87171', secondary: '#fff' } },
        }}
      />

      {/* Starfield */}
      <div className="star-layer stars-1" />
      <div className="star-layer stars-2" />
      {/* Ambient glow orbs */}
      <div className="bg-glow-1" />
      <div className="bg-glow-2" />
      {/* Dot grid */}
      <div className="bg-grid" />

      {/* AI Companion */}
      <AnimatePresence>
        {(view === 'input' || view === 'results') && (
          <RobotCompanion
            resumeUploaded={state.resumeUploaded}
            roleAnalyzed={state.roleAnalyzed}
            hasScore={state.score !== null}
            score={state.score}
            celebrationTick={state.celebrationTick}
            recommendations={state.recommendations}
            candidateName={state.candidateName}
            isLoading={isLoading}
            onQuickAction={state.handleRobotAction}
          />
        )}
      </AnimatePresence>

      {/* Page Router */}
      <AnimatePresence mode="wait">
        {view === 'landing' && (
          <LandingPage key="landing" onStart={() => setView('input')} />
        )}
        {view === 'input' && (
          <UploadView key="upload" state={state} />
        )}
        {view === 'analyzing' && (
          <AnalyzingView key="analyzing" state={state} />
        )}
        {view === 'results' && (
          <div key="results" className="results-shell">
            <ResultsView state={state} />
            <AiPanel state={state} />
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}

/* ──────────────────────────────
   AI Insights Side Panel
────────────────────────────── */
function AiPanel({ state }) {
  const { score, atsScore, insightTab, setInsightTab, recommendations } = state;

  const tabList = [
    { id: 'strategy', label: 'Strategy', icon: <Layout size={13} /> },
    { id: 'courses',  label: 'Academy',  icon: <BookOpen size={13} /> },
    { id: 'resume',   label: 'Resume',   icon: <FileEdit size={13} /> },
  ];

  // Determine score color using teal-based palette
  const scoreColor = score >= 80 ? 'var(--success)' : score >= 50 ? 'var(--accent)' : 'var(--danger)';

  return (
    <aside className="ai-panel">
      {/* Header */}
      <div className="ai-panel-head">
        <div className="kicker">
          <span className="kicker-dot" />
          AI Insights
        </div>
        <h2>Strategic Audit</h2>
        <p>Foundational analysis of your career trajectory.</p>
      </div>

      {/* Readiness */}
      <div className="readiness-row">
        <div className="readiness-cell">
          <span className="readiness-label">Market Readiness</span>
          <strong className="readiness-val" style={{ color: scoreColor }}>
            {score === null ? '—' : `${score}%`}
          </strong>
        </div>
        <div className="readiness-cell">
          <span className="readiness-label">ATS Optimization</span>
          <strong className="readiness-val">
            {atsScore === null ? '—' : `${atsScore}%`}
          </strong>
        </div>
      </div>

      {/* Tabs */}
      <div className="insight-tabs">
        {tabList.map(t => (
          <button
            key={t.id}
            className={`insight-tab${insightTab === t.id ? ' active' : ''}`}
            onClick={() => setInsightTab(t.id)}
          >
            {t.icon} {t.label}
          </button>
        ))}
      </div>

      {/* Body */}
      <div className="ai-panel-body">
        <AnimatePresence mode="wait">
          {insightTab === 'strategy' && (
            <motion.div
              key="strategy"
              initial={{ opacity: 0, x: 8 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -8 }}
              transition={{ duration: 0.25 }}
            >
              <div className="panel-section">
                <p className="panel-section-label"><TrendingUp size={11} /> Focus Architecture</p>
                <div className="focus-list">
                  {(recommendations?.focus_areas || ['Mapping target areas…']).map((item, i) => (
                    <div key={i} className="focus-card">
                      <span className="focus-idx">{i + 1}</span>
                      {item}
                    </div>
                  ))}
                </div>
              </div>

              <div className="panel-section">
                <p className="panel-section-label"><Zap size={11} /> Action Plan</p>
                <ul className="action-list">
                  {(recommendations?.action_plan || []).map((item, i) => (
                    <li key={i}>
                      <span className="action-dot" />
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </motion.div>
          )}

          {insightTab === 'courses' && (
            <motion.div
              key="courses"
              initial={{ opacity: 0, x: 8 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -8 }}
              transition={{ duration: 0.25 }}
            >
              <div className="panel-section">
                <p className="panel-section-label"><BookOpen size={11} /> Curated Academy</p>
                <div className="course-list">
                  {(recommendations?.courses || []).map((c, i) => (
                    <div key={i} className="course-card">
                      <div className="course-card-head">
                        <span className="course-platform">{c.platform}</span>
                        <span className="course-level">{c.level}</span>
                      </div>
                      <p className="course-title">{c.title}</p>
                      <span className="course-skill">Target: {c.for_skill}</span>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          )}

          {insightTab === 'resume' && (
            <motion.div
              key="resume"
              initial={{ opacity: 0, x: 8 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -8 }}
              transition={{ duration: 0.25 }}
            >
              <div className="panel-section">
                <p className="panel-section-label"><FileEdit size={11} /> Resume Improvements</p>
                <div className="resume-list">
                  {(recommendations?.resume_section_feedback || []).map((item, i) => (
                    <div key={i} className="resume-card">
                      <div className="resume-card-head">
                        <span className="resume-section-name">{item.section}</span>
                        <span className="resume-tag">UPGRADE</span>
                      </div>
                      <p className="resume-why">{item.why}</p>
                      <div className="resume-upgrade">{item.upgrade}</div>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </aside>
  );
}

export default App;
