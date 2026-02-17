import "./SkillGap.css";

function SkillGap({ score, matched, missing }) {
  let verdict = "";
  let color = "";
  const isCelebration = score >= 80;

  if (score >= 80) {
    verdict = "Strong Match";
    color = "#16a34a";
  } else if (score >= 60) {
    verdict = "Good Match";
    color = "#0f766e";
  } else if (score >= 40) {
    verdict = "Moderate Match";
    color = "#f59e0b";
  } else {
    verdict = "Needs Improvement";
    color = "#dc2626";
  }

  const totalTracked = matched.length + missing.length;
  const coverage = totalTracked
    ? Math.round((matched.length / totalTracked) * 100)
    : 0;
  const topMissing = missing.slice(0, 3);

  return (
    <div className={`skillgap ${isCelebration ? "celebration-mode" : ""}`}>
      {isCelebration && (
        <div className="confetti-field" aria-hidden="true">
          {Array.from({ length: 18 }).map((_, i) => (
            <span
              key={i}
              className="confetti-piece"
              style={{
                left: `${(i * 5.5) % 100}%`,
                animationDelay: `${(i % 6) * 0.12}s`,
              }}
            />
          ))}
        </div>
      )}

      <div className="skillgap-head">
        <h3>Skill Gap Analysis</h3>
        <p>Fit score for your selected role</p>
      </div>

      <div className="meter-wrap">
        <div className="meter-track">
          <div className="meter-fill" style={{ width: `${score}%` }} />
          <div className="meter-marker" style={{ left: `calc(${score}% - 10px)` }}>
            {score}%
          </div>
        </div>
      </div>

      <div className="score-box centered-score">
        <div
          className="score-circle"
          style={{
            background: `conic-gradient(
              ${color} ${score * 3.6}deg,
              rgba(203, 213, 225, 0.35) 0deg
            )`,
          }}
        >
          <span>{score}%</span>
        </div>

        <h2 style={{ color }}>{verdict}</h2>
        <p className="subtext">AI Match Score</p>
        {isCelebration && <p className="party-note">Party Popper Mode: Excellent Fit</p>}
      </div>

      <div className="summary-grid">
        <article className="summary-card">
          <p className="summary-label">Coverage</p>
          <strong>{coverage}%</strong>
        </article>
        <article className="summary-card">
          <p className="summary-label">Matched</p>
          <strong>{matched.length}</strong>
        </article>
        <article className="summary-card">
          <p className="summary-label">Missing</p>
          <strong>{missing.length}</strong>
        </article>
      </div>

      {topMissing.length > 0 && (
        <div className="priority-gaps">
          <h4>Priority Gaps</h4>
          <div className="priority-list">
            {topMissing.map((skill, i) => (
              <span key={`${skill}-${i}`}>{skill}</span>
            ))}
          </div>
        </div>
      )}

      <div className="gap-sections">
        <div className="gap-card success">
          <h4>Matched Skills</h4>
          {matched.length === 0 ? (
            <p className="empty">No strong matches found.</p>
          ) : (
            <ul>
              {matched.map((m, i) => (
                <li key={i}>
                  {m.job_skill} ({Math.round(m.similarity * 100)}%)
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="gap-card danger">
          <h4>Missing Skills</h4>
          {missing.length === 0 ? (
            <p className="empty">No critical gaps.</p>
          ) : (
            <ul>
              {missing.map((s, i) => (
                <li key={i}>{s}</li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}

export default SkillGap;
