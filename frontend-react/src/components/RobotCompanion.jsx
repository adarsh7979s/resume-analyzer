import { useEffect, useMemo, useState } from "react";
import "./RobotCompanion.css";

const RANDOM_MOODS = ["wave", "nod", "blink", "bounce"];

function getGuide(resumeUploaded, roleAnalyzed, hasScore, isLoading) {
  if (isLoading) {
    return {
      title: "Analyzing",
      tips: [
        "Reading your data and preparing suggestions.",
        "This usually takes a few seconds.",
      ],
    };
  }

  if (!resumeUploaded) {
    return {
      title: "Start Here",
      tips: [
        "Upload your resume PDF to begin.",
        "Use a clean one-page layout for better extraction.",
        "After upload, I will guide you to role analysis.",
      ],
    };
  }

  if (!roleAnalyzed) {
    return {
      title: "Next Step",
      tips: [
        "Enter a target role like AI Engineer.",
        "Try role variants like AIML Engineer too.",
        "Click Analyze Role to fetch required skills.",
      ],
    };
  }

  if (!hasScore) {
    return {
      title: "Almost Done",
      tips: [
        "Click Get Skill Gap to compute your fit score.",
        "Then focus on the Priority Gaps first.",
      ],
    };
  }

  return {
    title: "Great Progress",
    tips: [
      "Review missing skills and build targeted projects.",
      "Re-upload updated resume and compare scores.",
      "Aim for 80%+ before applying aggressively.",
    ],
  };
}

function RobotCompanion({ resumeUploaded, roleAnalyzed, hasScore, isLoading }) {
  const [mood, setMood] = useState("idle");
  const [tipIndex, setTipIndex] = useState(0);

  const guide = useMemo(
    () => getGuide(resumeUploaded, roleAnalyzed, hasScore, isLoading),
    [resumeUploaded, roleAnalyzed, hasScore, isLoading]
  );

  useEffect(() => {
    setTipIndex(0);
  }, [guide.title]);

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      const next = RANDOM_MOODS[Math.floor(Math.random() * RANDOM_MOODS.length)];
      setMood(next);

      window.setTimeout(() => {
        setMood("idle");
      }, 850);
    }, 2600 + Math.floor(Math.random() * 1800));

    return () => {
      window.clearInterval(intervalId);
    };
  }, []);

  useEffect(() => {
    const tipInterval = window.setInterval(() => {
      setTipIndex((prev) => (prev + 1) % guide.tips.length);
    }, 4200);

    return () => {
      window.clearInterval(tipInterval);
    };
  }, [guide.tips.length]);

  return (
    <aside className={`robot-companion robot-${mood} ${isLoading ? "robot-thinking" : ""}`}>
      <div className="robot-bubble">
        <p className="robot-title">{guide.title}</p>
        <p className="robot-tip">{guide.tips[tipIndex]}</p>
      </div>

      <div className="robot-shell" aria-hidden="true">
        <div className="robot-head">
          <span className="robot-eye" />
          <span className="robot-eye" />
          <span className="robot-mouth" />
        </div>
        <div className="robot-torso" />
        <span className="robot-arm robot-arm-left" />
        <span className="robot-arm robot-arm-right" />
      </div>
    </aside>
  );
}

export default RobotCompanion;
