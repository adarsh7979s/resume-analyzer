function SkillGap({ score, matched, missing }) {
  return (
    <div style={{ marginTop: "30px" }}>
      <h3>Skill Gap Analysis</h3>

      {/* Progress Circle */}
      <div
        style={{
          width: "140px",
          height: "140px",
          borderRadius: "50%",
          margin: "20px auto",
          background: `conic-gradient(#00c853 ${score * 3.6}deg, #ddd 0deg)`,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: "26px",
          fontWeight: "bold",
        }}
      >
        {score}%
      </div>

      <div style={{ display: "flex", gap: "40px", marginTop: "20px" }}>
        {/* Matched */}
        <div>
          <h4 style={{ color: "green" }}>✅ Matched Skills</h4>
          <ul>
            {matched.map((m, i) => (
              <li key={i}>
                {m.job_skill} ({Math.round(m.similarity * 100)}%)
              </li>
            ))}
          </ul>
        </div>

        {/* Missing */}
        <div>
          <h4 style={{ color: "red" }}>❌ Missing Skills</h4>
          <ul>
            {missing.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

export default SkillGap;
