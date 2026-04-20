import React from 'react';
import { motion } from 'framer-motion';
import { Clock, Target, TrendingUp, Trash2, RotateCcw } from 'lucide-react';

const item = {
  hidden:  { opacity: 0, y: 14 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.16, 1, 0.3, 1] } },
};

export default function HistoryView({ state }) {
  const { historyEntries, setActiveTab } = state;

  if (!historyEntries || historyEntries.length === 0) {
    return (
      <div className="hist-wrap">
        <div className="hist-header">
          <h2 className="hist-title">Analysis History</h2>
          <p className="hist-subtitle">Track your resume improvement over time</p>
        </div>
        <div className="hist-empty">
          <div className="hist-empty-icon">
            <Clock size={48} strokeWidth={1} />
          </div>
          <h3>No History Yet</h3>
          <p>Complete your first analysis to start tracking progress. Each analysis you run will appear here.</p>
          <button className="hist-start-btn" onClick={() => setActiveTab('dashboard')}>
            <Target size={14} /> Go to Dashboard
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="hist-wrap">
      <div className="hist-header">
        <h2 className="hist-title">Analysis History</h2>
        <p className="hist-subtitle">{historyEntries.length} analysis sessions recorded</p>
      </div>

      <div className="hist-list">
        {historyEntries.map((entry, i) => (
          <motion.div
            key={entry.id || i}
            className="hist-entry"
            variants={item}
            initial="hidden"
            animate="visible"
            transition={{ delay: i * 0.05 }}
          >
            <div className="hist-entry-left">
              <div className="hist-entry-score" style={{
                color: (entry.score || 0) >= 70 ? 'var(--success)' : 'var(--accent)',
              }}>
                {entry.score || '—'}%
              </div>
              <div className="hist-entry-info">
                <span className="hist-entry-role">{entry.role || 'Unknown Role'}</span>
                <span className="hist-entry-date">
                  {entry.date
                    ? new Date(entry.date).toLocaleDateString('en-US', {
                        month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
                      })
                    : 'Recent'}
                </span>
              </div>
            </div>
            <div className="hist-entry-right">
              <span className="hist-entry-skills">
                {entry.matched || 0} matched · {entry.missing || 0} gaps
              </span>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
