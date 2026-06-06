const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL
).replace(/\/+$/, "");


async function readErrorMessage(response, fallbackMessage) {
  try {
    const body = await response.json();

    if (typeof body?.detail === "string") {
      return body.detail;
    }

    if (Array.isArray(body?.detail)) {
      return body.detail
        .map((item) => item.msg || fallbackMessage)
        .join("；");
    }
  } catch {
    return fallbackMessage;
  }

  return fallbackMessage;
}


export async function startNarrativeAnalysis(
  projectId,
  {
    maxChunks = null,
    previousContextChars = 500,
    nextContextChars = 0,
    forceReanalyze = false,
    signal = undefined,
  } = {},
) {
  const response = await fetch(
    (
      `${API_BASE_URL}/api/projects/`
      + `${encodeURIComponent(projectId)}/narrative-analysis`
    ),
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        max_chunks: maxChunks,
        previous_context_chars: previousContextChars,
        next_context_chars: nextContextChars,
        force_reanalyze: forceReanalyze,
      }),
      signal,
    },
  );

  if (!response.ok) {
    throw new Error(
      await readErrorMessage(
        response,
        "启动 AI 分析失败。",
      ),
    );
  }

  return response.json();
}


export async function getNarrativeAnalysisRun(
  runId,
  { signal } = {},
) {
  const response = await fetch(
    `${API_BASE_URL}/api/narrative-analysis/${encodeURIComponent(runId)}`,
    {
      signal,
    },
  );

  if (!response.ok) {
    throw new Error(
      await readErrorMessage(
        response,
        "读取 AI 分析结果失败。",
      ),
    );
  }

  return response.json();
}
