import { useState } from "react";
import SkillGap from "./components/SkillGap";

const BASE_URL = "http://127.0.0.1:8000";

function App() {
  const [file, setFile] = useState(null);
  const [role, setRole] = useState("");

  const [status, setStatus] = useState("");
  const [resumeSkills, setResumeSkills] = useState([]);
  const [jobSkills, setJobSkills] = useState([]);

  const [score, setScore] = useState(null);
  const [matched, setMatched] = useState([]);
  const [missing, setMissing] = useState([]);

  /* ================= UPLOAD RESUME ================= */
  async function uploadResume() {
    if (!file) {
      setStatus("❌ Please select a PDF first");
      return;
    }

    setStatus("⏳ Uploading resume...");
    setResumeSkills([]);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${BASE_URL}/upload-resume`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (data.resume_skills_found) {
        setResumeSkills(data.resume_skills_found);
        setStatus("✅ Resume processed");
      } else {
        setStatus("❌ Resume processing failed");
      }
    } catch {
      setStatus("❌ Server error");
    }
  }

  /* ================= ANALYZE ROLE ================= */
  async function analyzeRole() {
    if (!role.trim()) {
      setStatus("❌ Enter a job role first");
      return;
    }

    setStatus("⏳ Analyzing role...");
    setJobSkills([]);

    try {
      const res = await fetch(`${BASE_URL}/analyze-role`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role }),
      });

      const data = await res.json();

      if (data.job_skills_required) {
        setJobSkills(data.job_skills_required);
        setStatus("✅ Role analyzed");
      } else {
        setStatus("❌ Role analysis failed");
      }
    } catch {
      setStatus("❌ Server error");
    }
  }

  /* ================= SKILL GAP ================= */
  async function getSkillGap() {
    setStatus("⏳ Calculating skill gap...");

    try {
      const res = await fetch(`${BASE_URL}/get-skill-gap`);
      const data = await res.json();

      if (data.match_score !== undefined) {
        setScore(data.match_score);
        setMatched(data.semantic_matches || []);
        setMissing(data.skills_missing || []);
        setStatus("✅ Skill gap calculated");
      } else {
        setStatus("❌ Failed to calculate skill gap");
      }
    } catch {
      setStatus("❌ Server error");
    }
  }

  return (
    <div style={{ padding: "30px", fontFamily: "Arial", maxWidth: "900px" }}>
      <h2>AI Resume & Skill Gap Analyzer</h2>

      {/* ============ UPLOAD RESUME ============ */}
      <h3>1️⃣ Upload Resume</h3>
      <input type="file" accept=".pdf" onChange={(e) => setFile(e.target.files[0])} />
      <br /><br />
      <button onClick={uploadResume}>Upload Resume</button>

      {resumeSkills.length > 0 && (
        <ul>
          {resumeSkills.map((s, i) => <li key={i}>{s}</li>)}
        </ul>
      )}

      <hr />

      {/* ============ ANALYZE ROLE ============ */}
      <h3>2️⃣ Analyze Job Role</h3>
      <input
        value={role}
        onChange={(e) => setRole(e.target.value)}
        placeholder="e.g. Game Developer"
      />
      <br /><br />
      <button onClick={analyzeRole}>Analyze Role</button>

      {jobSkills.length > 0 && (
        <ul>
          {jobSkills.map((s, i) => <li key={i}>{s}</li>)}
        </ul>
      )}

      <hr />

      {/* ============ SKILL GAP ============ */}
      <h3>3️⃣ Skill Gap</h3>
      <button onClick={getSkillGap}>Get Skill Gap</button>

      {/* ✅ THIS IS THE FIX */}
      {score !== null && (
        <SkillGap
          score={score}
          matched={matched}
          missing={missing}
        />
      )}

      {/* ============ STATUS ============ */}
      <p style={{ marginTop: "20px" }}>{status}</p>
    </div>
  );
}

export default App;
