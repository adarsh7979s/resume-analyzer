import { useEffect, useMemo, useRef, useState } from "react";
import SkillGap from "./components/SkillGap";
import RobotCompanion from "./components/RobotCompanion";
import "./App.css";

const BASE_URL = "http://127.0.0.1:8000";

function App() {
  const [isAnalysisMode, setIsAnalysisMode] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [activeStep, setActiveStep] = useState("");

  const [file, setFile] = useState(null);
  const [role, setRole] = useState("");
  const [resumeUploaded, setResumeUploaded] = useState(false);
  const [roleAnalyzed, setRoleAnalyzed] = useState(false);

  const [status, setStatus] = useState("");
  const [resumeSkills, setResumeSkills] = useState([]);
  const [jobSkills, setJobSkills] = useState([]);

  const [score, setScore] = useState(null);
  const [atsScore, setAtsScore] = useState(null);
  const [matched, setMatched] = useState([]);
  const [missing, setMissing] = useState([]);
  const [recommendations, setRecommendations] = useState(null);
  const [insightTab, setInsightTab] = useState("strategy");
  const [isDragActive, setIsDragActive] = useState(false);
  const [analysisId, setAnalysisId] = useState(null);
  const [celebrationTick, setCelebrationTick] = useState(0);

  const uploadSectionRef = useRef(null);
  const roleSectionRef = useRef(null);
  const gapSectionRef = useRef(null);
  const recommendationRef = useRef(null);

  useEffect(() => {
    const saved = window.sessionStorage.getItem("analysis_id");
    if (saved) {
      setAnalysisId(saved);
    }
  }, []);

  useEffect(() => {
    if (analysisId) {
      window.sessionStorage.setItem("analysis_id", analysisId);
    } else {
      window.sessionStorage.removeItem("analysis_id");
    }
  }, [analysisId]);

  async function parseJsonResponse(res) {
    let data = {};
    try {
      data = await res.json();
    } catch {
      data = {};
    }

    if (!res.ok || data.error) {
      const reason = data?.error || `Request failed (${res.status})`;
      throw new Error(reason);
    }

    return data;
  }

  function scrollTo(ref) {
    if (ref?.current) {
      ref.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  function handleDropFile(event) {
    event.preventDefault();
    setIsDragActive(false);

    const dropped = event.dataTransfer?.files?.[0];
    if (!dropped) {
      return;
    }
    setFile(dropped);
    setStatus(`Selected file: ${dropped.name}`);
  }

  async function uploadResume() {
    if (isLoading) {
      return;
    }

    setIsLoading(true);
    setActiveStep("upload");

    if (!file) {
      setStatus("Please select a PDF first.");
      setIsLoading(false);
      setActiveStep("");
      return;
    }

    if (file.type && file.type !== "application/pdf") {
      setStatus("Only PDF files are supported.");
      setIsLoading(false);
      setActiveStep("");
      return;
    }

    setStatus("Uploading resume...");

    setResumeSkills([]);
    setJobSkills([]);
    setScore(null);
    setAtsScore(null);
    setMatched([]);
    setMissing([]);
    setRecommendations(null);
    setRoleAnalyzed(false);
    setResumeUploaded(false);
    setAnalysisId(null);
    setIsAnalysisMode(false);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${BASE_URL}/upload-resume`, {
        method: "POST",
        body: formData,
      });

      const data = await parseJsonResponse(res);

      if (data.resume_skills_found) {
        if (data.analysis_id) {
          setAnalysisId(data.analysis_id);
        }
        setResumeSkills(data.resume_skills_found);
        setStatus("Resume processed.");

        setTimeout(() => {
          setResumeUploaded(true);
          scrollTo(roleSectionRef);
        }, 250);
      } else {
        setStatus("Resume processing failed.");
      }
    } catch (err) {
      setStatus(`Upload failed: ${err.message}`);
    }

    setIsLoading(false);
    setActiveStep("");
  }

  async function analyzeRole() {
    if (isLoading) {
      return;
    }

    setIsLoading(true);
    setActiveStep("role");

    if (!resumeUploaded) {
      setStatus("Please upload resume first.");
      setIsLoading(false);
      setActiveStep("");
      return;
    }

    if (!role.trim()) {
      setStatus("Enter a job role first.");
      setIsLoading(false);
      setActiveStep("");
      return;
    }

    if (!analysisId) {
      setStatus("Missing session. Please upload resume again.");
      setIsLoading(false);
      setActiveStep("");
      return;
    }

    setStatus("Analyzing role...");
    setJobSkills([]);
    setIsAnalysisMode(false);
    setScore(null);
    setAtsScore(null);
    setMatched([]);
    setMissing([]);
    setRecommendations(null);

    try {
      const res = await fetch(`${BASE_URL}/analyze-role`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          role,
          analysis_id: analysisId,
        }),
      });

      const data = await parseJsonResponse(res);

      if (data.job_skills_required) {
        if (data.analysis_id) {
          setAnalysisId(data.analysis_id);
        }
        setJobSkills(data.job_skills_required);
        setRoleAnalyzed(true);
        setStatus("Role analyzed.");
        setTimeout(() => scrollTo(gapSectionRef), 220);
      } else {
        setStatus("Role analysis failed.");
      }
    } catch (err) {
      setStatus(`Role analysis failed: ${err.message}`);
    }

    setIsLoading(false);
    setActiveStep("");
  }

  async function getSkillGap() {
    if (isLoading) {
      return;
    }

    setIsLoading(true);
    setActiveStep("gap");
    setStatus("Calculating skill gap...");

    try {
      if (!roleAnalyzed) {
        setStatus("Please analyze role first.");
        setIsLoading(false);
        setActiveStep("");
        return;
      }

      if (!analysisId) {
        setStatus("Session expired. Please upload resume again.");
        setIsLoading(false);
        setActiveStep("");
        return;
      }

      const res = await fetch(
        `${BASE_URL}/get-skill-gap?analysis_id=${encodeURIComponent(analysisId)}`
      );
      const data = await parseJsonResponse(res);

      if (data.match_score !== undefined) {
        setScore(data.match_score);
        setAtsScore(data.ats_score ?? data.match_score);
        setMatched(data.semantic_matches || []);
        setMissing(data.skills_missing || []);
        setRecommendations(data.recommendations || null);
        setInsightTab("strategy");
        setCelebrationTick((prev) => prev + 1);
        setStatus("Skill gap calculated.");
        setIsAnalysisMode(true);
        setTimeout(() => scrollTo(recommendationRef), 220);
      } else {
        setStatus("Failed to calculate skill gap.");
      }
    } catch (err) {
      setStatus(`Skill gap failed: ${err.message}`);
    }

    setIsLoading(false);
    setActiveStep("");
  }

  function handleRobotAction() {
    if (!resumeUploaded) {
      scrollTo(uploadSectionRef);
      return;
    }

    if (!roleAnalyzed) {
      scrollTo(roleSectionRef);
      if (!role.trim()) {
        setRole("AI Engineer");
        setStatus("Role template inserted. Press Analyze Role.");
      } else {
        analyzeRole();
      }
      return;
    }

    if (score === null) {
      scrollTo(gapSectionRef);
      getSkillGap();
      return;
    }

    scrollTo(recommendationRef);
  }

  const statusIsError = /failed|error|missing|expired|not found|unsupported/i.test(status);
  const statusClass = statusIsError ? "status status-error" : "status status-info";

  const completionPercent = useMemo(() => {
    if (score !== null) {
      return 100;
    }
    if (roleAnalyzed) {
      return 70;
    }
    if (resumeUploaded) {
      return 35;
    }
    return 0;
  }, [resumeUploaded, roleAnalyzed, score]);

  const readinessTone = score === null ? "neutral" : score >= 80 ? "strong" : score >= 60 ? "steady" : "risky";
  const resumeSkillCount = resumeSkills.length;
  const roleSkillCount = jobSkills.length;
  const missingCount = missing.length;

  return (
    <div className={`page ${isAnalysisMode ? "analysis-mode" : ""}`} aria-busy={isLoading}>
      <div className="bg-orb orb-one" />
      <div className="bg-orb orb-two" />
      <div className="bg-grid" />

      <RobotCompanion
        resumeUploaded={resumeUploaded}
        roleAnalyzed={roleAnalyzed}
        hasScore={score !== null}
        score={score}
        celebrationTick={celebrationTick}
        recommendations={recommendations}
        isLoading={isLoading}
        onQuickAction={handleRobotAction}
      />

      <div className="hero">
        <div className="content-rail">
          <header className="hero-header">
            <p className="hero-kicker">AI Career Copilot</p>
            <h1>Resume Skill Intelligence</h1>
            <p className="hero-subtitle">
              Upload your resume, analyze a target role, and get a tactical skill-gap roadmap.
            </p>

            <div className="progress-strip" aria-label="Workflow progress">
              <span className="progress-label">Workflow Completion</span>
              <div className="progress-track" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={completionPercent}>
                <div className="progress-fill" style={{ width: `${completionPercent}%` }} />
              </div>
              <span className="progress-value">{completionPercent}%</span>
            </div>

            <div className="kpi-row">
              <article className="kpi-card">
                <p>Resume Skills</p>
                <strong>{resumeSkillCount}</strong>
              </article>
              <article className="kpi-card">
                <p>Role Requirements</p>
                <strong>{roleSkillCount}</strong>
              </article>
              <article className="kpi-card">
                <p>ATS Score</p>
                <strong>{atsScore === null ? "-" : `${atsScore}%`}</strong>
              </article>
              <article className="kpi-card">
                <p>Critical Gaps</p>
                <strong>{score === null ? "-" : missingCount}</strong>
              </article>
            </div>
          </header>

          <section className="section-heading">
            <h2>Workflow</h2>
            <p>Complete the steps in order to generate accurate recommendations.</p>
          </section>

          <section ref={uploadSectionRef} className="card card-step">
            <h2>Step 1 Upload Resume</h2>
            <div className="form-rail">
              <div
                className={`upload-dropzone ${isDragActive ? "drag-active" : ""}`}
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragActive(true);
                }}
                onDragLeave={() => setIsDragActive(false)}
                onDrop={handleDropFile}
              >
                <input
                  id="resume-file-input"
                  className="file-input-hidden"
                  type="file"
                  accept=".pdf"
                  onChange={(e) => setFile(e.target.files[0])}
                  aria-describedby="upload-help status-region"
                />
                <p className="upload-dropzone-lead">Drop your resume here or choose a file.</p>
                <p className="upload-dropzone-sub">PDF only. Recommended max file size: 2MB.</p>
                <label htmlFor="resume-file-input" className="upload-dropzone-btn">
                  Upload Your Resume
                </label>
                <p className={`upload-dropzone-filename ${file ? "has-file" : ""}`}>
                  {file ? `Selected: ${file.name}` : "No file selected"}
                </p>
                <p className="upload-dropzone-privacy">Privacy guaranteed</p>
              </div>
              <p id="upload-help" className="sr-only">
                Upload a PDF resume to extract detected skills.
              </p>
              <button className="primary-btn" onClick={uploadResume} disabled={isLoading}>
                {isLoading && activeStep === "upload" ? "Uploading..." : "Upload Resume"}
              </button>
            </div>

            {resumeSkills.length > 0 && (
              <details className="skills-disclosure" open>
                <summary>Detected Resume Skills ({resumeSkills.length})</summary>
                <ul className="skill-list">
                  {resumeSkills.map((s, i) => (
                    <li key={i}>{s}</li>
                  ))}
                </ul>
              </details>
            )}
          </section>

          {resumeUploaded && (
            <section ref={roleSectionRef} className="card card-step">
              <h2>Step 2 Analyze Job Role</h2>
              <div className="form-rail role-rail">
                <label htmlFor="role-input" className="sr-only">
                  Target job role
                </label>
                <input
                  id="role-input"
                  value={role}
                  onChange={(e) => setRole(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !isLoading && resumeUploaded) {
                      analyzeRole();
                    }
                  }}
                  placeholder="e.g. AI Engineer"
                  aria-describedby="role-help status-region"
                />
                <p id="role-help" className="sr-only">
                  Enter the target role and analyze required skills.
                </p>
                <button
                  className="primary-btn"
                  onClick={analyzeRole}
                  disabled={!resumeUploaded || isLoading}
                >
                  {isLoading && activeStep === "role" ? "Analyzing..." : "Analyze Role"}
                </button>
              </div>

              {jobSkills.length > 0 && (
                <details className="skills-disclosure" open>
                  <summary>Role Requirements ({jobSkills.length})</summary>
                  <ul className="skill-list">
                    {jobSkills.map((s, i) => (
                      <li key={i}>{s}</li>
                    ))}
                  </ul>
                </details>
              )}
            </section>
          )}

          {roleAnalyzed && (
            <section ref={gapSectionRef} className="card card-step">
              <h2>Step 3 Run Skill Gap</h2>
              <div className="form-rail">
                <button className="primary-btn" onClick={getSkillGap} disabled={!roleAnalyzed || isLoading}>
                  {isLoading && activeStep === "gap" ? "Calculating..." : "Get Skill Gap"}
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

          {!isLoading && status && (
            <p
              id="status-region"
              className={statusClass}
              role={statusIsError ? "alert" : "status"}
              aria-live={statusIsError ? "assertive" : "polite"}
            >
              {status}
            </p>
          )}
        </div>

        {(roleAnalyzed || isAnalysisMode) && (
          <aside ref={recommendationRef} id="recommendation-panel" className="ai-panel">
            <section className="section-heading section-heading-panel">
              <h2>Insights</h2>
              <p>Use these recommendations to prioritize learning and projects.</p>
            </section>

            <div className={`readiness-card readiness-${readinessTone}`}>
              <p className="readiness-label">Readiness</p>
              <strong>{score === null ? "Pending" : `${score}%`}</strong>
              <p className="readiness-label">ATS Estimate</p>
              <strong>{atsScore === null ? "Pending" : `${atsScore}%`}</strong>
              <span>{score === null ? "Run Step 3 for final fit score" : recommendations?.summary || ""}</span>
            </div>

            <div className="insight-tabs" role="tablist" aria-label="Recommendation views">
              <button
                type="button"
                role="tab"
                aria-selected={insightTab === "strategy"}
                className={`insight-tab ${insightTab === "strategy" ? "active" : ""}`}
                onClick={() => setInsightTab("strategy")}
              >
                Strategy
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={insightTab === "courses"}
                className={`insight-tab ${insightTab === "courses" ? "active" : ""}`}
                onClick={() => setInsightTab("courses")}
              >
                Courses
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={insightTab === "resume"}
                className={`insight-tab ${insightTab === "resume" ? "active" : ""}`}
                onClick={() => setInsightTab("resume")}
              >
                Resume
              </button>
            </div>

            {insightTab === "strategy" && (
              <>
                <h3>Recommendation Blueprint</h3>

                <div className="ai-section">
                  <h4>Focus Areas</h4>
                  <div className="focus-grid">
                    {(recommendations?.focus_areas || ["Analyze skill gap to generate role-specific focus items."]).map((item, idx) => (
                      <article className="focus-card" key={`focus-${idx}`}>
                        <span className="focus-index">{idx + 1}</span>
                        <p>{item}</p>
                      </article>
                    ))}
                  </div>
                </div>

                <div className="ai-section">
                  <h4>Priority Gaps</h4>
                  <div className="chip-row">
                    {(recommendations?.priority_gaps || missing || []).slice(0, 6).map((item, idx) => (
                      <span key={`gap-chip-${idx}`} className="chip chip-gap">{item}</span>
                    ))}
                    {(!(recommendations?.priority_gaps || missing || []).length) && (
                      <span className="chip chip-ok">No high-priority gaps</span>
                    )}
                  </div>
                </div>

                <div className="ai-section">
                  <h4>Current Strengths</h4>
                  <div className="chip-row">
                    {(recommendations?.strengths || []).slice(0, 6).map((item, idx) => (
                      <span key={`strength-chip-${idx}`} className="chip chip-strength">{item}</span>
                    ))}
                    {(!(recommendations?.strengths || []).length) && (
                      <span className="chip chip-muted">Strengths will appear after scoring</span>
                    )}
                  </div>
                </div>

                <div className="ai-section">
                  <h4>Action Plan</h4>
                  <ul className="plan-list">
                    {(recommendations?.action_plan || ["Complete Step 3 to unlock a weekly action plan."]).map((item, idx) => (
                      <li key={`plan-${idx}`}>
                        <span className="plan-dot" aria-hidden="true" />
                        <span>{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </>
            )}

            {insightTab === "courses" && (
              <div className="ai-section">
                <h4>Suggested Courses</h4>
                <div className="course-list">
                  {(recommendations?.courses || []).slice(0, 6).map((course, idx) => (
                    <article className="course-card" key={`course-${idx}`}>
                      <p className="course-title">{course.title}</p>
                      <p className="course-meta">{course.platform} • {course.level}</p>
                      <span className="course-tag">For: {course.for_skill || "general"}</span>
                    </article>
                  ))}
                  {(!recommendations?.courses || recommendations.courses.length === 0) && (
                    <p className="course-empty">Run Step 3 to generate tailored course suggestions.</p>
                  )}
                </div>
              </div>
            )}

            {insightTab === "resume" && (
              <>
                <div className="ai-section">
                  <h4>Resume Upgrade</h4>
                  <ul className="plan-list">
                    {(recommendations?.resume_improvement_tips || [
                      "Run Step 3 to generate resume-specific improvement tips."
                    ]).map((item, idx) => (
                      <li key={`resume-tip-${idx}`}>
                        <span className="plan-dot" aria-hidden="true" />
                        <span>{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="ai-section">
                  <h4>What To Upgrade in Resume</h4>
                  <div className="resume-section-list">
                    {(recommendations?.resume_section_feedback || []).map((item, idx) => (
                      <article className="resume-section-card" key={`section-upgrade-${idx}`}>
                        <p className="resume-section-title">{item.section}</p>
                        <p className="resume-section-why">{item.why}</p>
                        <p className="resume-section-upgrade">{item.upgrade}</p>
                      </article>
                    ))}
                    {(!(recommendations?.resume_section_feedback || []).length) && (
                      <p className="course-empty">Run Step 3 to get section-wise resume upgrade suggestions.</p>
                    )}
                  </div>
                </div>
              </>
            )}
          </aside>
        )}
      </div>
    </div>
  );
}

export default App;
