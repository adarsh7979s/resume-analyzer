function resolveBaseUrl() {
  const configured = import.meta.env.VITE_API_BASE_URL?.trim();
  if (configured) {
    return configured.replace(/\/+$/, "");
  }

  if (typeof window !== "undefined") {
    return `${window.location.protocol}//${window.location.hostname}:8000`;
  }

  return "http://127.0.0.1:8000";
}

const BASE_URL = resolveBaseUrl();

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

export async function requestUploadResume(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${BASE_URL}/upload-resume`, {
    method: "POST",
    body: formData,
  });
  return parseJsonResponse(res);
}

export async function requestAnalyzeRole(role, analysisId) {
  const res = await fetch(`${BASE_URL}/analyze-role`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      role,
      analysis_id: analysisId,
    }),
  });
  return parseJsonResponse(res);
}

export async function requestSkillGap(analysisId) {
  const res = await fetch(
    `${BASE_URL}/get-skill-gap?analysis_id=${encodeURIComponent(analysisId)}`
  );
  return parseJsonResponse(res);
}

export async function requestHistory(analysisId) {
  const url = analysisId
    ? `${BASE_URL}/history?analysis_id=${encodeURIComponent(analysisId)}`
    : `${BASE_URL}/history`;
  const res = await fetch(url);
  return parseJsonResponse(res);
}
