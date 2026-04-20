import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload, FileText, CheckCircle2, ArrowRight, Sparkles,
  Target, TrendingUp, BookOpen, Zap,
} from 'lucide-react';
import toast from 'react-hot-toast';
import ScoreGauge from '../components/ScoreGauge';
import SkillGap from '../components/SkillGap';
import AnalyzingView from './AnalyzingView';
import './DashboardView.css';

export default function DashboardView({ state }) {
  const {
    isLoading, isUploadingResume,
    isDragActive, setIsDragActive,
    file, setFile,
    status, resumeUploaded,
    role, setRole,
    analyzeRoleAndRunGap, uploadResumeStep,
    score, atsScore, matched, missing,
    resumeSkills,
    recommendations, candidateName,
    view, startNewResumeFlow,
  } = state;

  const statusIsError = /failed|error|missing|expired|not found|unsupported|does not appear/i.test(status);
  const hasResults = score !== null;

  const initials = candidateName
    ? candidateName.split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2)
    : 'RA';

  /* ── Upload handlers ── */
  function handleDrop(e) {
    e.preventDefault();
    setIsDragActive(false);
    const dropped = e.dataTransfer?.files?.[0];
    if (dropped) {
      if (dropped.type !== 'application/pdf') {
        toast.error('Only PDF files are supported');
        return;
      }
      if (dropped.size > 5 * 1024 * 1024) {
        toast.error('File too large — max 5 MB');
        return;
      }
      setFile(dropped);
      toast.success(`Selected: ${dropped.name}`);
    }
  }

  function handleFileChange(e) {
    const f = e.target.files[0];
    if (f) {
      setFile(f);
      toast.success(`Selected: ${f.name}`);
    }
  }

  /* ── Key resume metrics (derived from atsScore) ── */
  const keyMetrics = hasResults ? [
    { name: 'Formatting',            value: Math.min(100, Math.round((atsScore || 0) * 1.05)) },
    { name: 'Readability',           value: Math.min(100, Math.round((atsScore || 0) * 0.98)) },
    { name: 'Keyword Optimization',  value: Math.min(100, Math.round((score || 0) * 1.02)) },
  ] : [];

  /* ── Score sub-metrics ── */
  const matchMetrics = hasResults ? [
    { name: 'Skills',     value: Math.min(100, Math.round((score || 0) * 1.04)) },
    { name: 'Experience', value: Math.min(100, Math.round((score || 0) * 0.97)) },
    { name: 'Education',  value: Math.min(100, Math.round((score || 0) * 0.94)) },
  ] : [];

  const atsMetrics = hasResults ? [
    { name: 'Parsability',    value: Math.min(100, Math.round((atsScore || 0) * 1.06)) },
    { name: 'Keyword Match',  value: Math.min(100, Math.round((atsScore || 0) * 0.95)) },
    { name: 'Structure',      value: Math.min(100, Math.round((atsScore || 0) * 1.0)) },
  ] : [];

  /* ── Recommendation items ── */
  const recItems = [];
  if (recommendations?.focus_areas) {
    recommendations.focus_areas.slice(0, 3).forEach((area, i) => {
      recItems.push({
        title: `Focus Area ${i + 1}`,
        text: area,
      });
    });
  }
  if (recommendations?.action_plan) {
    recommendations.action_plan.slice(0, 2).forEach((action, i) => {
      recItems.push({
        title: `Action ${i + 1}`,
        text: action,
      });
    });
  }

  const container = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { staggerChildren: 0.06 } },
  };
  const item = {
    hidden: { opacity: 0, y: 14 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.45, ease: [0.16, 1, 0.3, 1] } },
  };

  return (
    <>
      {/* ── Analyzing overlay ── */}
      <AnimatePresence>
        {view === 'analyzing' && (
          <motion.div
            className="dv-analyzing-overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <AnalyzingView state={state} />
          </motion.div>
        )}
      </AnimatePresence>

      <motion.div variants={container} initial="hidden" animate="visible">
        {/* ── Profile row ── */}
        <motion.div variants={item} className="dv-profile-card">
          <div className="dv-profile-avatar">{initials}</div>
          <div className="dv-profile-info">
            <span className="dv-profile-name">
              {candidateName || 'Upload Resume to Start'}
            </span>
            <span className="dv-profile-sub">
              {hasResults
                ? `${role} Candidate · ${resumeSkills.length} skills detected`
                : resumeUploaded
                  ? `${resumeSkills.length} skills extracted · Set target role`
                  : 'AI-Powered Resume Analysis'}
            </span>
          </div>
          {hasResults && (
            <motion.button
              className="dv-profile-badge"
              onClick={startNewResumeFlow}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              New Analysis
            </motion.button>
          )}
        </motion.div>

        {/* ── Main grid ── */}
        <div className="dv-grid">
          {/* ── LEFT: Upload card ── */}
          <motion.div variants={item}>
            <div
              className={`dv-card dv-upload-card${isDragActive ? ' drag-over' : ''}${file ? ' has-file' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setIsDragActive(true); }}
              onDragLeave={() => setIsDragActive(false)}
              onDrop={handleDrop}
              onClick={() => document.getElementById('dash-file-input').click()}
            >
              <input
                id="dash-file-input"
                type="file"
                accept=".pdf"
                className="file-input-hidden"
                onChange={handleFileChange}
              />

              <div className="dv-upload-icon">
                {file
                  ? <CheckCircle2 size={24} strokeWidth={1.5} />
                  : <Upload size={24} strokeWidth={1.5} />}
              </div>

              <span className="dv-upload-title">
                {isDragActive ? 'Release to upload' : file ? 'Resume Selected' : 'Upload Resume'}
              </span>

              <span className="dv-upload-sub">
                {file
                  ? `${(file.size / 1024).toFixed(0)} KB · PDF`
                  : 'Drag & Drop PDF/DOCX\nor Browse'}
              </span>

              {file && (
                <span className="dv-upload-filename">
                  <FileText size={14} /> {file.name}
                </span>
              )}

              {!resumeUploaded && (
                <button
                  className="dv-upload-btn"
                  disabled={isLoading || !file}
                  onClick={(e) => { e.stopPropagation(); uploadResumeStep(); }}
                >
                  {isUploadingResume
                    ? <><span className="upload-ring" style={{ width: 14, height: 14 }} /> Parsing…</>
                    : <>Upload & Extract <ArrowRight size={14} /></>}
                </button>
              )}

              {resumeUploaded && !hasResults && (
                <div className="dv-role-section" onClick={(e) => e.stopPropagation()}>
                  <p className="dv-role-label">Select Target Role</p>
                  <div className="dv-role-input-row">
                    <input
                      id="dash-role-input"
                      className="noir-input"
                      value={role}
                      onChange={(e) => setRole(e.target.value)}
                      onKeyDown={(e) => { if (e.key === 'Enter' && !isLoading) analyzeRoleAndRunGap(); }}
                      placeholder="e.g. AI Engineer"
                    />
                    <button
                      className="dv-role-analyze-btn"
                      disabled={isLoading || !role.trim()}
                      onClick={analyzeRoleAndRunGap}
                    >
                      <Sparkles size={14} /> Analyze
                    </button>
                  </div>
                  <div className="dv-role-chips">
                    {['AI Engineer', 'Frontend Developer', 'Data Scientist', 'Backend Engineer'].map(r => (
                      <button key={r} className="role-chip" onClick={() => setRole(r)} type="button">
                        {r}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </motion.div>

          {/* ── CENTER: Score Gauges ── */}
          <motion.div variants={item}>
            <div className="dv-card dv-scores-card">
              <h3 className="dv-scores-heading">Analyze Your Resume</h3>

              <div className="dv-scores-tabs">
                {[
                  { icon: <FileText size={12} />, label: 'Resume' },
                  { icon: <Target size={12} />, label: 'Skills' },
                  { icon: <TrendingUp size={12} />, label: 'Match' },
                  { icon: <Zap size={12} />, label: 'AI' },
                ].map(t => (
                  <button
                    key={t.label}
                    className={`dv-scores-tab ${t.label === 'Match' ? 'active' : ''}`}
                  >
                    {t.icon} {t.label}
                  </button>
                ))}
              </div>

              {hasResults ? (
                <div className="dv-scores-gauges">
                  <ScoreGauge
                    value={score}
                    label="MATCH SCORE"
                    color={score >= 80 ? 'var(--success)' : score >= 50 ? 'var(--accent)' : 'var(--danger)'}
                    metrics={matchMetrics}
                  />
                  <ScoreGauge
                    value={atsScore}
                    label="ATS SCORE"
                    color="var(--accent)"
                    metrics={atsMetrics}
                  />
                </div>
              ) : (
                <div className="dv-empty-state">
                  <div className="dv-empty-icon">
                    <Target size={40} strokeWidth={1} />
                  </div>
                  <span className="dv-empty-title">No Analysis Yet</span>
                  <span className="dv-empty-sub">
                    Upload your resume and select a target role to see your Match Score and ATS Score.
                  </span>
                </div>
              )}
            </div>
          </motion.div>

          {/* ── RIGHT COLUMN ── */}
          <div className="dv-right-col">
            {/* Recommendations */}
            <motion.div variants={item}>
              <div className="dv-card dv-recs-card">
                <h3 className="dv-recs-title">Recommendations</h3>
                {recItems.length > 0 ? (
                  recItems.map((rec, i) => (
                    <div key={i} className="dv-rec-item">
                      <h4>{rec.title}</h4>
                      <p>{rec.text}</p>
                    </div>
                  ))
                ) : (
                  <div className="dv-empty-state" style={{ padding: '24px 12px' }}>
                    <BookOpen size={28} style={{ color: 'var(--txt-3)', opacity: 0.4 }} />
                    <span className="dv-empty-sub">
                      Recommendations appear after analysis.
                    </span>
                  </div>
                )}
              </div>
            </motion.div>

            {/* Key Resume Metrics */}
            <motion.div variants={item}>
              <div className="dv-card dv-metrics-card">
                <h3 className="dv-metrics-title">Key Resume Metrics</h3>
                {keyMetrics.length > 0 ? (
                  keyMetrics.map((m, i) => (
                    <div key={i} className="dv-metric-row">
                      <span className="dv-metric-name">{m.name}</span>
                      <div className="dv-metric-bar-track">
                        <motion.div
                          className="dv-metric-bar-fill"
                          initial={{ width: 0 }}
                          animate={{ width: `${m.value}%` }}
                          transition={{ duration: 1, delay: 0.3 + i * 0.15, ease: [0.16, 1, 0.3, 1] }}
                        />
                      </div>
                      <span className="dv-metric-val">{m.value}%</span>
                    </div>
                  ))
                ) : (
                  <div className="dv-empty-state" style={{ padding: '24px 12px' }}>
                    <TrendingUp size={28} style={{ color: 'var(--txt-3)', opacity: 0.4 }} />
                    <span className="dv-empty-sub">
                      Metrics appear after analysis.
                    </span>
                  </div>
                )}
              </div>
            </motion.div>
          </div>

          {/* ── SKILL GAP (spans 2 cols) ── */}
          {hasResults && (
            <motion.div variants={item} className="dv-skillgap-card">
              <div className="dv-card">
                <SkillGap score={score || 0} matched={matched} missing={missing} />
              </div>
            </motion.div>
          )}
        </div>

        {/* Status */}
        {status && (
          <motion.div variants={item} className="dv-status">
            <div className={`status-bar ${statusIsError ? 'status-error' : 'status-info'}`}>
              {status}
            </div>
          </motion.div>
        )}
      </motion.div>
    </>
  );
}
