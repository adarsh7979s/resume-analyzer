import { useState, useRef, useEffect } from "react";
import {
  requestUploadResume,
  requestAnalyzeRole,
  requestSkillGap,
  requestHistory,
} from "../services/api";

const ANALYSIS_MIN_DURATION_MS = 3200;
const sleep = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));

export function useAnalyzer() {
  const [view, setView] = useState("landing");
  const [isAnalysisMode, setIsAnalysisMode] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("dashboard");

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
    // Restore history from localStorage
    try {
      const h = localStorage.getItem('ra_history');
      if (h) setHistoryEntries(JSON.parse(h));
    } catch {}
  }, []);

  useEffect(() => {
    if (analysisId) {
      window.sessionStorage.setItem("analysis_id", analysisId);
    } else {
      window.sessionStorage.removeItem("analysis_id");
    }
  }, [analysisId]);

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
    if (isLoading) return;

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
      const uploadData = await requestUploadResume(file);

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
    if (isLoading) return;
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
      const roleData = await requestAnalyzeRole(role, analysisId);
      if (activeAnalysisRunRef.current !== runId) return;

      if (!roleData.job_skills_required) {
        throw new Error("Role analysis failed");
      }
      setJobSkills(roleData.job_skills_required || []);

      await sleep(300);
      if (activeAnalysisRunRef.current !== runId) return;

      setAnalysisProgress(2);
      setStatus("Calculating skill gap...");
      const gapData = await requestSkillGap(analysisId);
      if (activeAnalysisRunRef.current !== runId) return;

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
        if (activeAnalysisRunRef.current !== runId) return;
      }

      // Save local history entry
      setHistoryEntries(prev => {
        const entry = {
          id: Date.now(),
          role: role,
          score: gapData.match_score,
          atsScore: gapData.ats_score ?? gapData.match_score,
          matched: (gapData.semantic_matches || []).length,
          missing: (gapData.skills_missing || []).length,
          date: new Date().toISOString(),
        };
        const next = [entry, ...prev].slice(0, 20);
        try { localStorage.setItem('ra_history', JSON.stringify(next)); } catch {}
        return next;
      });

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

    if (insightTab === "strategy") return setInsightTab("courses");
    if (insightTab === "courses") return setInsightTab("resume");
    setInsightTab("strategy");
  }

  return {
    view, setView,
    isAnalysisMode,
    isLoading,
    activeTab, setActiveTab,
    file, setFile,
    role, setRole,
    isDragActive, setIsDragActive,
    status, setStatus,
    analysisProgress,
    isUploadingResume,
    resumeSkills, jobSkills,
    score, atsScore,
    matched, missing,
    recommendations,
    candidateName,
    historyEntries,
    insightTab, setInsightTab,
    celebrationTick,
    resumeUploaded,
    roleAnalyzed,
    startNewResumeFlow,
    uploadResumeStep,
    analyzeRoleAndRunGap,
    handleRobotAction,
  };
}
