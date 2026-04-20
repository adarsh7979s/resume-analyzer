import React, { useEffect } from 'react';
import { AnimatePresence } from 'framer-motion';
import { Toaster } from 'react-hot-toast';
import confetti from 'canvas-confetti';
import './App.css';
import { useAnalyzer } from './hooks/useAnalyzer';
import LandingPage from './components/LandingPage';
import DashboardLayout from './components/DashboardLayout';
import DashboardView from './pages/DashboardView';
import InsightsView from './pages/InsightsView';
import HistoryView from './pages/HistoryView';
import AiChatbot from './components/AiChatbot';
import '../src/pages/InsightsView.css';
import '../src/pages/HistoryView.css';

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

/* ── Settings placeholder ── */
function SettingsPage() {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      justifyContent: 'center', gap: 16, padding: '80px 24px',
      textAlign: 'center',
    }}>
      <div style={{
        width: 64, height: 64, borderRadius: 16,
        background: 'rgba(20,184,166,0.08)',
        border: '1px solid rgba(20,184,166,0.2)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: 'var(--accent)', fontSize: '1.5rem',
      }}>
        ⚙️
      </div>
      <h2 style={{
        fontFamily: 'var(--font-head)', fontWeight: 800,
        fontSize: '1.5rem', color: 'var(--txt)',
      }}>
        Settings
      </h2>
      <p style={{
        fontSize: '0.95rem', color: 'var(--txt-3)',
        maxWidth: 400, lineHeight: 1.6,
      }}>
        Configure your analysis preferences, notification settings, and integrations.
      </p>
      <span style={{
        padding: '6px 16px', background: 'var(--accent-dim)',
        border: '1px solid var(--border-teal)', borderRadius: 'var(--r-full)',
        fontSize: '0.72rem', fontWeight: 700, color: 'var(--accent)',
        letterSpacing: '0.06em',
      }}>
        COMING SOON
      </span>
    </div>
  );
}

function App() {
  const state = useAnalyzer();
  const { view, setView, activeTab, setActiveTab } = state;

  // Fire confetti when results come in with a good score
  useConfetti(view === 'results' || state.score !== null ? state.score : null);

  /* ── Determine what the sidebar active tab renders ── */
  function renderTabContent() {
    switch (activeTab) {
      case 'dashboard':
        return <DashboardView state={state} />;
      case 'analyzer':
        // Analyzer tab redirects to dashboard with focus
        return <DashboardView state={state} />;
      case 'insights':
        return <InsightsView state={state} />;
      case 'history':
        return <HistoryView state={state} />;
      case 'settings':
        return <SettingsPage />;
      default:
        return <DashboardView state={state} />;
    }
  }

  return (
    <>
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

      <AnimatePresence mode="wait">
        {view === 'landing' ? (
          <div className="page" key="landing">
            {/* Starfield */}
            <div className="star-layer stars-1" />
            <div className="star-layer stars-2" />
            <div className="bg-glow-1" />
            <div className="bg-glow-2" />
            <div className="bg-grid" />

            <LandingPage onStart={() => setView('input')} />
          </div>
        ) : (
          <DashboardLayout
            key="dashboard"
            activeTab={activeTab}
            setActiveTab={setActiveTab}
            candidateName={state.candidateName}
          >
            {renderTabContent()}
          </DashboardLayout>
        )}
      </AnimatePresence>

      {/* AI Chatbot — visible in dashboard */}
      {view !== 'landing' && <AiChatbot state={state} />}
    </>
  );
}

export default App;
