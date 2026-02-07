import { useState } from "react";
import SkillGap from "./components/SkillGap";
import "./App.css";

const BASE_URL = "http://127.0.0.1:8000";

function App() {

  const [isLoading, setIsLoading] = useState(false);

  const [file, setFile] = useState(null);
  const [role, setRole] = useState("");
  const [resumeUploaded, setResumeUploaded] = useState(false);
  const [roleAnalyzed, setRoleAnalyzed] = useState(false);

  const [status, setStatus] = useState("");
  const [resumeSkills, setResumeSkills] = useState([]);
  const [jobSkills, setJobSkills] = useState([]);

  const [score, setScore] = useState(null);
  const [matched, setMatched] = useState([]);
  const [missing, setMissing] = useState([]);

  /* ================= UPLOAD RESUME ================= */
  async function uploadResume() {
    setIsLoading(true);

    if (!file) {
      setStatus("❌ Please select a PDF first");
      return;
    }

    setStatus("⏳ Uploading resume...");

    // 🔹 clear old data (VERY IMPORTANT)
    setResumeSkills([]);
    setJobSkills([]);
    setScore(null);
    setMatched([]);
    setMissing([]);
    setRoleAnalyzed(false);


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

        setTimeout(() => {
          setResumeUploaded(true);
        }, 600);

      }else {
        setStatus("❌ Resume processing failed");
      }
    } catch {
      setStatus("❌ Server error");
    }
    setIsLoading(false);

  }

  /* ================= ANALYZE ROLE ================= */
  async function analyzeRole() {
    setIsLoading(true);

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
        setRoleAnalyzed(true);
        setStatus("✅ Role analyzed");
      }else {
        setStatus("❌ Role analysis failed");
      }
    } catch {
      setStatus("❌ Server error");
    }
    setIsLoading(false);

  }

  /* ================= SKILL GAP ================= */
  async function getSkillGap() {
    setIsLoading(true);

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
    setIsLoading(false);

  }

  return (
    <div className="page">
      <div className="hero">

        {/* ===== HERO HEADER ===== */}
        <header className="hero-header">
          <h1>AI Resume & Skill Gap Analyzer</h1>
          <p className="hero-subtitle">
            Upload your resume and instantly see how well you match a job role.
          </p>
        </header>

        {/* ===== STEP 1 ===== */}
        <section className="card">
          <h3>1️⃣ Upload Resume</h3>

          <input
            type="file"
            accept=".pdf"
            onChange={(e) => setFile(e.target.files[0])}
          />

          <button onClick={uploadResume}>Upload Resume</button>

          {resumeSkills.length > 0 && (
            <ul className="skill-list">
              {resumeSkills.map((s, i) => (
                <li key={i}>{s}</li>
              ))}
            </ul>
          )}
        </section>

        {/* ===== STEP 2 ===== */}
        {resumeUploaded && (
          <section className="card">
            <h3>2️⃣ Analyze Job Role</h3>

            <input
              value={role}
              onChange={(e) => setRole(e.target.value)}
              placeholder="e.g. Game Developer"
            />

            <button onClick={analyzeRole} disabled={!resumeUploaded}>
              Analyze Role
            </button>

            {jobSkills.length > 0 && (
              <ul className="skill-list">
                {jobSkills.map((s, i) => (
                  <li key={i}>{s}</li>
                ))}
              </ul>
            )}
          </section>
        )}


        {/* ===== STEP 3 ===== */}
        {roleAnalyzed && (
          <section className="card">
            <h3>3️⃣ Skill Gap</h3>

            <button onClick={getSkillGap} disabled={!roleAnalyzed}>
              Get Skill Gap
            </button>

            {score !== null && (
              <SkillGap
                score={score}
                matched={matched}
                missing={missing}
              />
            )}
          </section>
        )}

        {/* ===== STATUS ===== */}
        {isLoading && (
          <div className="ai-loader">
            <div className="dot" />
            <div className="dot" />
            <div className="dot" />
            <span>AI is analyzing</span>
          </div>
        )}

        {!isLoading && status !== "" && <p className="status">{status}</p>}


      </div>
    </div>
  );
}

export default App;
