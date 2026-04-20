import { useEffect, useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './RobotCompanion.css';

const RANDOM_MOODS = ['wave', 'nod', 'blink', 'bounce', 'tilt'];

function getGuide(resumeUploaded, roleAnalyzed, hasScore, isLoading, score, isCelebrating, recommendations, candidateName) {
  const name = candidateName?.trim() || '';
  const hi = name ? `Hi ${name}` : 'Hey there';

  if (isCelebrating) return {
    title: 'Excellent!',
    tips: [`${hi} — ${score}% match! Outstanding result.`, 'Great momentum to start applying now.', 'Review your matched strengths and leverage them.'],
  };
  if (isLoading) return {
    title: 'Analyzing',
    tips: [`${hi}, parsing your resume now.`, 'Give me a moment to evaluate your role fit.'],
  };
  if (!resumeUploaded) return {
    title: 'Start Here',
    tips: [`${hi}, upload your resume PDF to begin.`, 'Use a clean, single-column layout for best extraction.', 'After upload, set your target role in Step 2.'],
  };
  if (!roleAnalyzed) return {
    title: 'Set Target',
    tips: [`${hi} — great upload!`, 'Enter a role like "Backend Engineer" or "Data Scientist".', 'I will benchmark your skills against real job requirements.'],
  };
  if (!hasScore) return {
    title: 'Almost Done',
    tips: [`${hi}, click Analyze to generate your fit score.`, 'I will surface your top strengths and critical gaps.', 'Then we build your action plan together.'],
  };
  if (recommendations) {
    const tips = [];
    if (recommendations.summary) tips.push(recommendations.summary);
    if (Array.isArray(recommendations.focus_areas)) tips.push(...recommendations.focus_areas.slice(0, 2));
    if (Array.isArray(recommendations.action_plan)) tips.push(...recommendations.action_plan.slice(0, 1));
    if (Array.isArray(recommendations.courses) && recommendations.courses.length > 0) {
      const c = recommendations.courses[0];
      tips.push(`Start with: "${c.title}" on ${c.platform}.`);
    }
    return { title: 'Personal Guide', tips: tips.length ? tips : ['Focus on closing your highest-priority skill gap first.'] };
  }
  return { title: 'Great Progress', tips: ['Build projects targeting missing skills.', 'Re-upload an updated resume to track improvement.', 'Aim for 80%+ before applying aggressively.'] };
}

export default function RobotCompanion({ resumeUploaded, roleAnalyzed, hasScore, score, celebrationTick, recommendations, candidateName, isLoading, onQuickAction }) {
  const [mood, setMood]         = useState('idle');
  const [tipIndex, setTipIndex] = useState(0);
  const [collapsed, setCollapsed] = useState(false);
  const [isCelebrating, setIsCelebrating] = useState(false);

  const guide = useMemo(
    () => getGuide(resumeUploaded, roleAnalyzed, hasScore, isLoading, score, isCelebrating, recommendations, candidateName),
    [resumeUploaded, roleAnalyzed, hasScore, isLoading, score, isCelebrating, recommendations, candidateName]
  );

  // Mood animation loop
  useEffect(() => {
    const id = setInterval(() => {
      const next = RANDOM_MOODS[Math.floor(Math.random() * RANDOM_MOODS.length)];
      setMood(next);
      setTimeout(() => setMood('idle'), 1200);
    }, 4500);
    return () => clearInterval(id);
  }, []);

  // Tip rotation
  useEffect(() => {
    const id = setInterval(() => setTipIndex(p => (p + 1) % (guide.tips?.length || 1)), 6000);
    return () => clearInterval(id);
  }, [guide.tips?.length]);

  // Celebration
  useEffect(() => {
    if (typeof score !== 'number' || score < 80) return;
    setIsCelebrating(true);
    const t = setTimeout(() => setIsCelebrating(false), 8000);
    return () => clearTimeout(t);
  }, [score, celebrationTick]);

  const currentTip = guide.tips?.[tipIndex % guide.tips.length] ?? 'Ready to help you land your next role.';
  const tipCount   = guide.tips?.length ?? 1;

  const actionLabel = !resumeUploaded ? 'UPLOAD' : !roleAnalyzed ? 'SET ROLE' : !hasScore ? 'ANALYZE' : 'INSIGHTS';

  const faceClass = isCelebrating ? 'face-happy'
    : isLoading ? 'face-thinking'
    : mood === 'blink' ? 'face-blink'
    : 'face-neutral';

  return (
    <motion.aside
      className={`robot-companion robot-${mood} ${isCelebrating ? 'robot-celebrate' : ''}`}
      drag
      dragMomentum={false}
      dragConstraints={{ left: -window.innerWidth + 280, right: 0, top: -window.innerHeight + 200, bottom: 0 }}
      initial={{ opacity: 0, scale: 0.85, y: 20 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.85, y: 20 }}
      transition={{ type: 'spring', stiffness: 280, damping: 22 }}
    >
      {/* Avatar */}
      <div className="robot-shell">
        <div className="robot-ant" />
        <div className={`robot-head ${faceClass}`}>
          <div className="robot-eye"><span className="robot-pupil" /></div>
          <div className="robot-eye"><span className="robot-pupil" /></div>
        </div>
      </div>

      {/* Bubble */}
      <motion.div
        className="robot-bubble"
        animate={{ scaleX: collapsed ? 0.92 : 1, scaleY: collapsed ? 0.1 : 1, opacity: collapsed ? 0 : 1 }}
        transition={{ type: 'spring', stiffness: 320, damping: 28 }}
        style={{ pointerEvents: collapsed ? 'none' : 'auto' }}
      >
        <div className="robot-bubble-top">
          <span className="robot-drag-handle">
            {isCelebrating ? 'CELEBRATION' : guide.title.toUpperCase()}
          </span>
          <button type="button" className="robot-icon-btn" onClick={() => setCollapsed(c => !c)}>
            −
          </button>
        </div>

        <div className="robot-bubble-content">
          <AnimatePresence mode="wait">
            <motion.p
              key={currentTip}
              className="robot-tip"
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.3 }}
            >
              {currentTip}
            </motion.p>
          </AnimatePresence>

          <div className="robot-actions">
            <button
              type="button"
              className="robot-action-btn robot-action-main"
              onClick={onQuickAction}
            >
              {actionLabel}
            </button>
            <button
              type="button"
              className="robot-action-btn"
              onClick={() => setTipIndex(p => (p + 1) % tipCount)}
            >
              SKIP
            </button>
          </div>
        </div>
      </motion.div>

      {/* Collapsed toggle */}
      {collapsed && (
        <motion.button
          type="button"
          className="robot-icon-btn"
          style={{ marginTop: 4, width: 28, height: 28, fontSize: '0.8rem' }}
          onClick={() => setCollapsed(false)}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          +
        </motion.button>
      )}
    </motion.aside>
  );
}
