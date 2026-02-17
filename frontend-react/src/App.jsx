import { useState } from "react";
import SkillGap from "./components/SkillGap";
import RobotCompanion from "./components/RobotCompanion";
import "./App.css";

const BASE_URL = "http://127.0.0.1:8000";

function App() {
  const [isAnalysisMode, setIsAnalysisMode] = useState(false);
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

  async function uploadResume() {
    setIsLoading(true);

    if (!file) {
      setStatus("Please select a PDF first.");
      setIsLoading(false);
      return;
    }

    setStatus("Uploading resume...");

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
        setStatus("Resume processed.");

        setTimeout(() => {
          setResumeUploaded(true);
        }, 350);
      } else {
        setStatus("Resume processing failed.");
      }
    } catch {
      setStatus("Server error.");
    }

    setIsLoading(false);
  }

  async function analyzeRole() {
    setIsLoading(true);

    if (!role.trim()) {
      setStatus("Enter a job role first.");
      setIsLoading(false);
      return;
    }

    setStatus("Analyzing role...");
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
        setStatus("Role analyzed.");
      } else {
        setStatus("Role analysis failed.");
      }
    } catch {
      setStatus("Server error.");
    }

    setIsLoading(false);
  }

  async function getSkillGap() {
    setIsLoading(true);
    setStatus("Calculating skill gap...");

    try {
      const res = await fetch(`${BASE_URL}/get-skill-gap`);
      const data = await res.json();

      if (data.match_score !== undefined) {
        setScore(data.match_score);
        setMatched(data.semantic_matches || []);
        setMissing(data.skills_missing || []);
        setStatus("Skill gap calculated.");
        setIsAnalysisMode(true);
      } else {
        setStatus("Failed to calculate skill gap.");
      }
    } catch {
      setStatus("Server error.");
    }

    setIsLoading(false);
  }

  return (
    <div className={`page ${isAnalysisMode ? "analysis-mode" : ""}`}>
      <div className="bg-orb orb-one" />
      <div className="bg-orb orb-two" />
      <div className="bg-grid" />
      <RobotCompanion
        resumeUploaded={resumeUploaded}
        roleAnalyzed={roleAnalyzed}
        hasScore={score !== null}
        isLoading={isLoading}
      />

      <div className="hero">
        <div className="content-rail">
          <header className="hero-header">
            <h1>Resume Skill Intelligence</h1>
            <p className="hero-subtitle">
              Upload your resume and measure real fit for a role in three quick steps.
            </p>
          </header>

          <section className="card card-step">
            <h3>Step 1 Upload Resume</h3>
            <div className="form-rail">
              <input
                type="file"
                accept=".pdf"
                onChange={(e) => setFile(e.target.files[0])}
              />
              <button onClick={uploadResume}>Upload Resume</button>
            </div>

            {resumeSkills.length > 0 && (
              <ul className="skill-list">
                {resumeSkills.map((s, i) => (
                  <li key={i}>{s}</li>
                ))}
              </ul>
            )}
          </section>

          {resumeUploaded && (
            <section className="card card-step">
              <h3>Step 2 Analyze Job Role</h3>
              <div className="form-rail role-rail">
                <input
                  value={role}
                  onChange={(e) => setRole(e.target.value)}
                  placeholder="e.g. AI Engineer"
                />
                <button onClick={analyzeRole} disabled={!resumeUploaded}>
                  Analyze Role
                </button>
              </div>

              {jobSkills.length > 0 && (
                <ul className="skill-list">
                  {jobSkills.map((s, i) => (
                    <li key={i}>{s}</li>
                  ))}
                </ul>
              )}
            </section>
          )}

          {roleAnalyzed && (
            <section className="card card-step">
              <h3>Step 3 Run Skill Gap</h3>
              <div className="form-rail">
                <button onClick={getSkillGap} disabled={!roleAnalyzed}>
                  Get Skill Gap
                </button>
              </div>
            </section>
          )}

          {score !== null && (
            <section className="result-card">
              <SkillGap score={score} matched={matched} missing={missing} />
            </section>
          )}

          {isLoading && (
            <div className="ai-loader">
              <div className="dot" />
              <div className="dot" />
              <div className="dot" />
              <span>AI is analyzing</span>
            </div>
          )}

          {!isLoading && status && <p className="status">{status}</p>}
        </div>

        {isAnalysisMode && (
          <aside className="ai-panel">
            <h3>Recommendations</h3>

            <div className="ai-section">
              <h4>Focus Areas</h4>
              <ul>
                <li>Strengthen missing core skills</li>
                <li>Improve domain-specific knowledge</li>
                <li>Practice applied projects</li>
              </ul>
            </div>

            <div className="ai-section">
              <h4>Suggested Learning</h4>
              <ul>
                <li>Hands-on courses</li>
                <li>Project-based learning</li>
                <li>Interview-focused preparation</li>
              </ul>
            </div>
          </aside>
        )}
      </div>
    </div>
  );
}

export default App;
