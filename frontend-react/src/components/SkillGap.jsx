function SkillGap({ score, matched, missing }) {
  return (
    <div className="skillgap">
      <h3>Skill Gap Analysis</h3>

      {/* ===== SCORE VERDICT ===== */}
      <div className="score-box">
        <div
          className="score-circle"
          style={{
            background: `conic-gradient(
              #22c55e ${score * 3.6}deg,
              rgba(229, 231, 235, 0.6) 0deg
            )`,
          }}
        >
          <span>{score}%</span>
        </div>
        <p>AI Match Score</p>
      </div>

      {/* ===== DETAILS ===== */}
      <div className="gap-sections">
        <div className="gap-card success">
          <h4>✔ Matched Skills</h4>
          {matched.length === 0 ? (
            <p className="empty">No strong matches found</p>
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
          <h4>✘ Missing Skills</h4>
          {missing.length === 0 ? (
            <p className="empty">No critical gaps 🎉</p>
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
