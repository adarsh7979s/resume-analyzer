import { useEffect, useMemo, useState } from "react";
import "./RobotCompanion.css";

const RANDOM_MOODS = ["wave", "nod", "blink", "bounce", "tilt"];

function getGuide(
  resumeUploaded,
  roleAnalyzed,
  hasScore,
  isLoading,
  score,
  isCelebrating,
  recommendations
) {
  if (isCelebrating) {
    return {
      title: "Party Mode",
      tips: [
        `Score ${score}%! Outstanding match.`,
        "I am doing a happy dance across the screen.",
        "Keep this momentum and start applying now.",
      ],
    };
  }

  if (isLoading) {
    return {
      title: "Analyzing",
      tips: [
        "I am parsing your resume and evaluating role fit.",
        "Give me a few seconds and I will return your score.",
      ],
    };
  }

  if (!resumeUploaded) {
    return {
      title: "Start Here",
      tips: [
        "Upload your resume PDF to begin.",
        "Use a clean single-column layout for better extraction.",
        "After upload, enter your target role in Step 2.",
      ],
    };
  }

  if (!roleAnalyzed) {
    return {
      title: "Next Step",
      tips: [
        "Enter a role like Backend Engineer or AI Engineer.",
        "Click Analyze Role to load required skills.",
        "I will compare those skills against your resume.",
      ],
    };
  }

  if (!hasScore) {
    return {
      title: "Almost Done",
      tips: [
        "Run Step 3 to generate your final fit score.",
        "Then focus first on priority gaps.",
      ],
    };
  }

  if (recommendations) {
    const dynamicTips = [];
    if (recommendations.summary) {
      dynamicTips.push(recommendations.summary);
    }
    if (Array.isArray(recommendations.focus_areas)) {
      dynamicTips.push(...recommendations.focus_areas.slice(0, 2));
    }
    if (Array.isArray(recommendations.action_plan)) {
      dynamicTips.push(...recommendations.action_plan.slice(0, 1));
    }
    if (Array.isArray(recommendations.courses) && recommendations.courses.length > 0) {
      const c = recommendations.courses[0];
      dynamicTips.push(`Start with course: ${c.title} on ${c.platform}.`);
    }

    return {
      title: "Personal Guidance",
      tips: dynamicTips.length
        ? dynamicTips
        : ["Follow your recommendation panel and close the highest-priority gap first."],
    };
  }

  return {
    title: "Great Progress",
    tips: [
      "Focus on missing skills and build targeted projects.",
      "Re-upload your updated resume and compare improvements.",
      "Aim for 80%+ before applying aggressively.",
    ],
  };
}

function RobotCompanion({
  resumeUploaded,
  roleAnalyzed,
  hasScore,
  score,
  celebrationTick,
  recommendations,
  isLoading,
  onQuickAction,
}) {
  const [mood, setMood] = useState("idle");
  const [tipIndex, setTipIndex] = useState(0);
  const [collapsed, setCollapsed] = useState(false);
  const [isCelebrating, setIsCelebrating] = useState(false);
  const [position, setPosition] = useState(() => {
    try {
      const saved = window.localStorage.getItem("robot_position");
      if (!saved) {
        return { x: null, y: null };
      }
      const parsed = JSON.parse(saved);
      if (typeof parsed?.x === "number" && typeof parsed?.y === "number") {
        return { x: parsed.x, y: parsed.y };
      }
    } catch {
      // Ignore storage errors and fall back to default CSS position.
    }
    return { x: null, y: null };
  });

  const guide = useMemo(
    () =>
      getGuide(
        resumeUploaded,
        roleAnalyzed,
        hasScore,
        isLoading,
        score,
        isCelebrating,
        recommendations
      ),
    [resumeUploaded, roleAnalyzed, hasScore, isLoading, score, isCelebrating, recommendations]
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
      }, 900);
    }, 2300 + Math.floor(Math.random() * 2000));

    return () => window.clearInterval(intervalId);
  }, []);

  useEffect(() => {
    const tipInterval = window.setInterval(() => {
      setTipIndex((prev) => (prev + 1) % guide.tips.length);
    }, 4600);

    return () => window.clearInterval(tipInterval);
  }, [guide.tips.length]);

  useEffect(() => {
    if (position.x === null || position.y === null) {
      return;
    }
    window.localStorage.setItem("robot_position", JSON.stringify(position));
  }, [position]);

  useEffect(() => {
    if (typeof score !== "number" || score < 80) {
      return;
    }
    setIsCelebrating(true);
    setMood("bounce");
    setCollapsed(false);

    const timer = window.setTimeout(() => {
      setIsCelebrating(false);
      setMood("idle");
    }, 5000);

    return () => window.clearTimeout(timer);
  }, [score, celebrationTick]);

  function handleDragStart(event) {
    if (event.button !== undefined && event.button !== 0) {
      return;
    }

    const node = event.currentTarget.closest(".robot-companion");
    if (!node) {
      return;
    }

    event.preventDefault();
    const rect = node.getBoundingClientRect();
    const offsetX = event.clientX - rect.left;
    const offsetY = event.clientY - rect.top;

    function onMove(moveEvent) {
      const maxX = Math.max(window.innerWidth - rect.width, 0);
      const maxY = Math.max(window.innerHeight - rect.height, 0);
      const nextX = Math.min(Math.max(moveEvent.clientX - offsetX, 0), maxX);
      const nextY = Math.min(Math.max(moveEvent.clientY - offsetY, 0), maxY);
      setPosition({ x: nextX, y: nextY });
    }

    function onEnd() {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onEnd);
    }

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onEnd);
  }

  const currentTip = guide.tips[tipIndex];
  const quickActionLabel = !resumeUploaded
    ? "Go To Step 1"
    : !roleAnalyzed
      ? "Next: Analyze Role"
      : !hasScore
        ? "Run Skill Gap"
        : "Open Recommendations";
  const faceClass = isCelebrating
    ? "face-happy"
    : isLoading
      ? "face-thinking"
      : mood === "blink"
        ? "face-blink"
        : "face-neutral";

  return (
    <aside
      className={`robot-companion robot-${mood} ${isLoading ? "robot-thinking" : ""} ${
        isCelebrating ? "robot-celebrate" : ""
      } ${collapsed ? "robot-collapsed" : ""}`}
      style={
        !isCelebrating && position.x !== null && position.y !== null
          ? {
              left: `${position.x}px`,
              top: `${position.y}px`,
              right: "auto",
              bottom: "auto",
            }
          : undefined
      }
    >
      <div className="robot-shell" aria-hidden="true">
        <div className="robot-glow" />
        <div className="robot-ant" />
        <div className={`robot-head ${faceClass}`}>
          <span className="robot-brow robot-brow-left" />
          <span className="robot-brow robot-brow-right" />

          <div className="robot-eye robot-eye-left">
            <span className="robot-pupil" />
          </div>
          <div className="robot-eye robot-eye-right">
            <span className="robot-pupil" />
          </div>

          <span className="robot-cheek robot-cheek-left" />
          <span className="robot-cheek robot-cheek-right" />
          <span className="robot-mouth" />
          <span className="robot-spark robot-spark-left">*</span>
          <span className="robot-spark robot-spark-right">*</span>
        </div>

        <div className="robot-torso">
          <span className="robot-core" />
        </div>
        <span className="robot-arm robot-arm-left" />
        <span className="robot-arm robot-arm-right" />
      </div>

      <div className="robot-bubble" role="status" aria-live="polite">
        <div className="robot-bubble-top">
          <button
            type="button"
            className="robot-drag-handle"
            onPointerDown={handleDragStart}
            aria-label="Drag assistant"
            title="Drag assistant"
          >
            {isCelebrating ? "Party Mode" : guide.title}
          </button>
          <button
            type="button"
            className="robot-icon-btn"
            onClick={() => setCollapsed((v) => !v)}
            aria-label={collapsed ? "Expand assistant" : "Minimize assistant"}
          >
            {collapsed ? "+" : "-"}
          </button>
        </div>

        {!collapsed && (
          <>
            <p className="robot-tip">{currentTip}</p>
            <div className="robot-actions">
              <button
                type="button"
                className="robot-action-btn robot-action-main"
                onClick={onQuickAction}
                disabled={isLoading}
              >
                {quickActionLabel}
              </button>
              <button
                type="button"
                className="robot-action-btn"
                onClick={() => setTipIndex((prev) => (prev + 1) % guide.tips.length)}
              >
                Next Tip
              </button>
            </div>
          </>
        )}
      </div>
    </aside>
  );
}

export default RobotCompanion;
