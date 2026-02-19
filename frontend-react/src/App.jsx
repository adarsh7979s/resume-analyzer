import { useEffect, useMemo, useRef, useState } from "react";
import SkillGap from "./components/SkillGap";
import RobotCompanion from "./components/RobotCompanion";
import "./App.css";

const BASE_URL = "http://127.0.0.1:8000";

const ANALYSIS_STEPS = [
  "Mapping role requirements",
  "Matching resume against role",
  "Generating recommendations",
];
const ANALYSIS_MIN_DURATION_MS = 3200;

const sleep = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));

function App() {
  const [view, setView] = useState("input");
  const [isAnalysisMode, setIsAnalysisMode] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const [file, setFile] = useState(null);
  const [role, setRole] = useState("");
  const [isDragActive, setIsDragActive] = useState(false);

  const [status, setStatus] = useState("");
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [isUploadingResume, setIsUploadingResume] = useState(false);

  const [resumeSkills, setResumeSkills] = useState([]);
  const [jobSkills, setJobSkills] = useState([]);
  const [score, setScore] = useState(null);
  const [atsScore, setAtsScore] = useState(null);
  const [matched, setMatched] = useState([]);
  const [missing, setMissing] = useState([]);
  const [recommendations, setRecommendations] = useState(null);
  const [candidateName, setCandidateName] = useState("");
  const [historyEntries, setHistoryEntries] = useState([]);
  const [insightTab, setInsightTab] = useState("strategy");
  const [analysisId, setAnalysisId] = useState(null);
  const [celebrationTick, setCelebrationTick] = useState(0);
  const activeAnalysisRunRef = useRef(0);
  const resumeUploaded = resumeSkills.length > 0 && Boolean(analysisId);
  const roleAnalyzed = jobSkills.length > 0;

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

  async function requestUploadResume() {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(`${BASE_URL}/upload-resume`, {
      method: "POST",
      body: formData,
    });
    return parseJsonResponse(res);
  }

  async function requestAnalyzeRole(nextAnalysisId) {
    const res = await fetch(`${BASE_URL}/analyze-role`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        role,
        analysis_id: nextAnalysisId,
      }),
    });
    return parseJsonResponse(res);
  }

  async function requestSkillGap(nextAnalysisId) {
    const res = await fetch(
      `${BASE_URL}/get-skill-gap?analysis_id=${encodeURIComponent(nextAnalysisId)}`
    );
    return parseJsonResponse(res);
  }

  async function requestHistory(nextAnalysisId) {
    const url = nextAnalysisId
      ? `${BASE_URL}/history?analysis_id=${encodeURIComponent(nextAnalysisId)}`
      : `${BASE_URL}/history`;
    const res = await fetch(url);
    return parseJsonResponse(res);
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

  function startNewResumeFlow() {
    activeAnalysisRunRef.current += 1;
    setIsLoading(false);
    setIsUploadingResume(false);
    setIsAnalysisMode(false);
    setView("input");
    setAnalysisProgress(0);
    setFile(null);
    setRole("");
    setResumeSkills([]);
    setJobSkills([]);
    setScore(null);
    setAtsScore(null);
    setMatched([]);
    setMissing([]);
    setRecommendations(null);
    setHistoryEntries([]);
    setCandidateName("");
    setInsightTab("strategy");
    setAnalysisId(null);
    setStatus("Upload a new resume to start a fresh analysis.");
  }

  async function uploadResumeStep() {
    if (isLoading) {
      return;
    }

    if (!file) {
      setStatus("Please select a PDF first.");
      return;
    }

    if (file.type && file.type !== "application/pdf") {
      setStatus("Only PDF files are supported.");
      return;
    }

    setIsLoading(true);
    setIsUploadingResume(true);
    setStatus("Uploading and parsing resume...");

    setResumeSkills([]);
    setAnalysisId(null);
    setJobSkills([]);
    setScore(null);
    setAtsScore(null);
    setMatched([]);
    setMissing([]);
    setRecommendations(null);
    setHistoryEntries([]);
    setCandidateName("");
    setInsightTab("strategy");
    setIsAnalysisMode(false);
    setView("input");

    try {
      const uploadData = await requestUploadResume();

      if (!uploadData.resume_skills_found || !uploadData.analysis_id) {
        throw new Error("Resume processing failed");
      }

      setAnalysisId(uploadData.analysis_id);
      setResumeSkills(uploadData.resume_skills_found || []);
      setCandidateName(uploadData?.personal_details?.name || "");
      setStatus("Resume uploaded. Please enter your target job role.");
    } catch (err) {
      setStatus(`Analysis failed: ${err.message}`);
      setView("input");
    } finally {
      setIsUploadingResume(false);
      setIsLoading(false);
    }
  }

  async function analyzeRoleAndRunGap() {
    if (isLoading) {
      return;
    }

    if (!resumeUploaded || !analysisId) {
      setStatus("Please upload resume first.");
      return;
    }

    if (!role.trim()) {
      setStatus("Please enter the target job role.");
      return;
    }

    setIsLoading(true);
    setIsAnalysisMode(false);
    setView("analyzing");
    setAnalysisProgress(0);
    setStatus("Starting analysis...");
    const analysisStartTs = Date.now();
    const runId = Date.now();
    activeAnalysisRunRef.current = runId;

    setJobSkills([]);
    setScore(null);
    setAtsScore(null);
    setMatched([]);
    setMissing([]);
    setRecommendations(null);

    try {
      setAnalysisProgress(1);
      setStatus("Analyzing target role...");
      const roleData = await requestAnalyzeRole(analysisId);
      if (activeAnalysisRunRef.current !== runId) {
        return;
      }

      if (!roleData.job_skills_required) {
        throw new Error("Role analysis failed");
      }

      setJobSkills(roleData.job_skills_required || []);

      await sleep(300);
      if (activeAnalysisRunRef.current !== runId) {
        return;
      }
      setAnalysisProgress(2);
      setStatus("Calculating skill gap...");
      const gapData = await requestSkillGap(analysisId);
      if (activeAnalysisRunRef.current !== runId) {
        return;
      }

      if (gapData.match_score === undefined) {
        throw new Error("Skill gap calculation failed");
      }

      setScore(gapData.match_score);
      setAtsScore(gapData.ats_score ?? gapData.match_score);
      setMatched(gapData.semantic_matches || []);
      setMissing(gapData.skills_missing || []);
      setRecommendations(gapData.recommendations || null);
      setInsightTab("strategy");

      try {
        const historyData = await requestHistory(analysisId);
        if (activeAnalysisRunRef.current === runId) {
          const nextHistory = Array.isArray(historyData.history) ? historyData.history : [];
          setHistoryEntries(nextHistory.slice().reverse().slice(0, 5));
        }
      } catch {
        if (activeAnalysisRunRef.current === runId) {
          setHistoryEntries([]);
        }
      }

      setAnalysisProgress(3);
      const elapsed = Date.now() - analysisStartTs;
      if (elapsed < ANALYSIS_MIN_DURATION_MS) {
        await sleep(ANALYSIS_MIN_DURATION_MS - elapsed);
        if (activeAnalysisRunRef.current !== runId) {
          return;
        }
      }
      setCelebrationTick((prev) => prev + 1);
      setIsAnalysisMode(true);
      setView("results");
      setStatus("Analysis complete.");
    } catch (err) {
      setStatus(`Analysis failed: ${err.message}`);
      setView("input");
    } finally {
      setIsLoading(false);
    }
  }

  function handleRobotAction() {
    if (view === "input") {
      if (!resumeUploaded) {
        setStatus("Upload your resume first.");
        return;
      }
      if (!role.trim()) {
        setRole("AI Engineer");
        setStatus("Target role filled. Click Analyze Role.");
        return;
      }
      analyzeRoleAndRunGap();
      return;
    }

    if (insightTab === "strategy") {
      setInsightTab("courses");
      return;
    }
    if (insightTab === "courses") {
      setInsightTab("resume");
      return;
    }
    setInsightTab("strategy");
  }

  const statusIsError = /failed|error|missing|expired|not found|unsupported/i.test(status);
  const statusClass = statusIsError ? "status status-error" : "status status-info";

  const readinessTone =
    score === null ? "neutral" : score >= 80 ? "strong" : score >= 60 ? "steady" : "risky";
  const resumeSkillCount = resumeSkills.length;
  const roleSkillCount = jobSkills.length;
  const missingCount = missing.length;

  const completionPercent = useMemo(() => {
    if (view === "results") {
      return 100;
    }
    if (view === "analyzing") {
      return Math.min(95, analysisProgress * 24);
    }
    return 0;
  }, [view, analysisProgress]);

  return (
    <div className={`page ${isAnalysisMode ? "analysis-mode" : ""}`} aria-busy={isLoading}>
      <div className="bg-orb orb-one" />
      <div className="bg-orb orb-two" />
      <div className="bg-grid" />

      {(view === "input" || view === "results") && (
        <RobotCompanion
          resumeUploaded={resumeUploaded}
          roleAnalyzed={roleAnalyzed}
          hasScore={score !== null}
          score={score}
          celebrationTick={celebrationTick}
          recommendations={recommendations}
          candidateName={candidateName}
          isLoading={isLoading}
          onQuickAction={handleRobotAction}
        />
      )}

      {view === "input" && (
        <main className="stage-shell input-shell">
          <section className="input-hero-card">
            <p className="hero-kicker">Resume Checker</p>
            <h1>Measure Role Fit. Improve Fast. Get Interview-Ready.</h1>
            <p className="hero-subtitle">
              Upload your resume first. Then enter your target role to run full analysis.
            </p>

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
              {!resumeUploaded && !isUploadingResume && (
                <label htmlFor="resume-file-input" className="upload-dropzone-btn">
                  Choose Resume File
                </label>
              )}
              {isUploadingResume && (
                <div className="upload-inline-loader" aria-live="polite">
                  <span className="upload-inline-ring" aria-hidden="true" />
                  <p>We are scanning your file...</p>
                </div>
              )}
              <p className={`upload-dropzone-filename ${file ? "has-file" : ""}`}>
                {file ? `Selected: ${file.name}` : "No file selected"}
              </p>
              <p className="upload-dropzone-privacy">Privacy guaranteed</p>
            </div>

            {!resumeUploaded && (
              <button className="primary-btn start-analysis-btn" onClick={uploadResumeStep} disabled={isLoading}>
                {isLoading ? "Uploading..." : "Upload Resume"}
              </button>
            )}

            {resumeUploaded && (
              <>
                <details className="skills-disclosure input-skills-block" open>
                  <summary>Detected Resume Skills ({resumeSkills.length})</summary>
                  <ul className="skill-list">
                    {resumeSkills.map((s, i) => (
                      <li key={`input-resume-${i}`}>{s}</li>
                    ))}
                  </ul>
                </details>

                <div className="target-role-block">
                  <p className="role-instruction">Please enter the target job role.</p>
                  <label htmlFor="role-input" className="target-role-label">
                    Target Job Role
                  </label>
                  <input
                    id="role-input"
                    value={role}
                    onChange={(e) => setRole(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !isLoading) {
                        analyzeRoleAndRunGap();
                      }
                    }}
                    placeholder="e.g. AI Engineer"
                    aria-describedby="role-help status-region"
                  />
                  <p id="role-help" className="sr-only">
                    Enter the target role for skill-gap analysis.
                  </p>
                </div>

                <button className="primary-btn start-analysis-btn" onClick={analyzeRoleAndRunGap} disabled={isLoading}>
                  Analyze Role
                </button>
              </>
            )}

            <p id="upload-help" className="sr-only">
              Upload a PDF resume to extract detected skills.
            </p>

            {status && (
              <p
                id="status-region"
                className={statusClass}
                role={statusIsError ? "alert" : "status"}
                aria-live={statusIsError ? "assertive" : "polite"}
              >
                {status}
              </p>
            )}
          </section>
        </main>
      )}

      {view === "analyzing" && (
        <main className="stage-shell scanning-shell">
          <section className="scan-hero-card">
            <div className="scan-hero-top">
              <p className="hero-kicker">Analysis in progress</p>
              <button type="button" className="scan-new-upload-btn" onClick={startNewResumeFlow}>
                Upload New Resume
              </button>
            </div>
            <h1>We are scanning your file.</h1>
            <p className="hero-subtitle">Please wait while we parse, evaluate, and prepare recommendations.</p>

            <div className="scan-loader-wrap">
              <div className="scan-loader-ring" aria-hidden="true" />
            </div>

            <div className="scan-progress-track" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={completionPercent}>
              <div className="scan-progress-fill" style={{ width: `${completionPercent}%` }} />
            </div>
          </section>

          <section className="scan-grid">
            <article className="scan-score-card">
              <h3>Your Score</h3>
              <div className="scan-score-gauge" aria-hidden="true">
                <span className="scan-score-needle" />
              </div>
              <div className="scan-score-lines" aria-hidden="true">
                <span />
                <span />
              </div>
              <ul>
                <li><span>Content</span><em /></li>
                <li><span>Section</span><em /></li>
                <li><span>ATS Essentials</span><em /></li>
                <li><span>Tailoring</span><em /></li>
              </ul>
            </article>

            <article className="scan-steps-card">
              {ANALYSIS_STEPS.map((step, idx) => {
                const order = idx + 1;
                const completed = analysisProgress > order;
                const active = analysisProgress === order;
                return (
                  <div
                    key={step}
                    className={`scan-step-row ${completed ? "is-complete" : ""} ${active ? "is-active" : ""}`}
                  >
                    <span className="scan-step-icon" aria-hidden="true">v</span>
                    <p>{step}</p>
                  </div>
                );
              })}
            </article>
          </section>
        </main>
      )}

      {view === "results" && (
        <>
          <div className="hero">
            <div className="content-rail">
              <header className="hero-header">
                <div className="results-topbar">
                  <p className="hero-kicker">AI Career Copilot</p>
                  <button type="button" className="scan-new-upload-btn" onClick={startNewResumeFlow}>
                    Upload New Resume
                  </button>
                </div>
                <h1>Resume Skill Intelligence</h1>
                <p className="hero-subtitle">
                  Analysis complete. Review your fit score, ATS estimate, and targeted improvement plan.
                </p>

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

              <section className="result-card">
                <SkillGap score={score || 0} matched={matched} missing={missing} />
              </section>

              <section className="card card-step compact-card">
                <h2>Detected Skills</h2>
                <details className="skills-disclosure" open>
                  <summary>Resume Skills ({resumeSkills.length})</summary>
                  <ul className="skill-list">
                    {resumeSkills.map((s, i) => (
                      <li key={`resume-${i}`}>{s}</li>
                    ))}
                  </ul>
                </details>
                <details className="skills-disclosure">
                  <summary>Role Requirements ({jobSkills.length})</summary>
                  <ul className="skill-list">
                    {jobSkills.map((s, i) => (
                      <li key={`role-${i}`}>{s}</li>
                    ))}
                  </ul>
                </details>
              </section>

              <section className="card card-step compact-card">
                <h2>Recent Analyses</h2>
                {historyEntries.length === 0 ? (
                  <p className="history-empty">No analysis history yet.</p>
                ) : (
                  <ul className="history-list">
                    {historyEntries.map((item, i) => {
                      const roleLabel = item.current_role || item.role || "Unknown role";
                      const matchScore = item.match_score ?? "-";
                      const ats = item.ats_score ?? "-";
                      const when = item.updated_at || item.created_at || "";
                      return (
                        <li key={`history-${i}`} className="history-item">
                          <p className="history-role">{roleLabel}</p>
                          <p className="history-metrics">Match: {matchScore}% • ATS: {ats}%</p>
                          {when ? <p className="history-time">{new Date(when).toLocaleString()}</p> : null}
                        </li>
                      );
                    })}
                  </ul>
                )}
              </section>

              {status && (
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

            <aside id="recommendation-panel" className="ai-panel">
              <section className="section-heading section-heading-panel">
                <h2>Insights</h2>
                <p>Use these recommendations to prioritize learning and projects.</p>
              </section>

              <div className={`readiness-card readiness-${readinessTone}`}>
                <p className="readiness-label">Readiness</p>
                <strong>{score === null ? "Pending" : `${score}%`}</strong>
                <p className="readiness-label">ATS Estimate</p>
                <strong>{atsScore === null ? "Pending" : `${atsScore}%`}</strong>
                <span>{recommendations?.summary || "Recommendations unavailable."}</span>
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
                      {(recommendations?.focus_areas || ["No focus areas available."]).map((item, idx) => (
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
                    </div>
                  </div>

                  <div className="ai-section">
                    <h4>Action Plan</h4>
                    <ul className="plan-list">
                      {(recommendations?.action_plan || []).map((item, idx) => (
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
                  </div>
                </div>
              )}

              {insightTab === "resume" && (
                <>
                  <div className="ai-section">
                    <h4>Resume Upgrade</h4>
                    <ul className="plan-list">
                      {(recommendations?.resume_improvement_tips || []).map((item, idx) => (
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
                    </div>
                  </div>
                </>
              )}
            </aside>
          </div>
        </>
      )}
    </div>
  );
}

export default App;



