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

    if (body?.detail && typeof body.detail === "object") {
      return body.detail.message || fallbackMessage;
    }
  } catch {
    return fallbackMessage;
  }

  return fallbackMessage;
}


export async function startScriptGeneration(
  projectId,
  {
    maxChunks = null,
    chunkIds = [],
    generationStyle = "standard",
    adaptationMode = "faithful",
    provider = "deepseek",
    model = "deepseek-v4-pro",
    thinkingEnabled = false,
    signal = undefined,
  } = {},
) {
  const response = await fetch(
    (
      `${API_BASE_URL}/api/projects/`
      + `${encodeURIComponent(projectId)}/script-generation`
    ),
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        max_chunks: maxChunks,
        chunk_ids: chunkIds,
        generation_style: generationStyle,
        adaptation_mode: adaptationMode,
        provider,
        model,
        thinking_enabled: thinkingEnabled,
      }),
      signal,
    },
  );

  if (!response.ok) {
    throw new Error(
      await readErrorMessage(
        response,
        "启动剧本生成失败。",
      ),
    );
  }

  return response.json();
}


export async function getScriptGenerationRun(
  runId,
  {
    includeUnits = true,
    signal,
  } = {},
) {
  const query = new URLSearchParams({
    include_units: includeUnits ? "true" : "false",
  });
  const response = await fetch(
    (
      `${API_BASE_URL}/api/script-generation/`
      + `${encodeURIComponent(runId)}?${query.toString()}`
    ),
    {
      signal,
    },
  );

  if (!response.ok) {
    throw new Error(
      await readErrorMessage(
        response,
        "读取剧本生成进度失败。",
      ),
    );
  }

  return response.json();
}


export async function getLatestScriptGenerationRun(
  projectId,
  { signal } = {},
) {
  const response = await fetch(
    (
      `${API_BASE_URL}/api/projects/`
      + `${encodeURIComponent(projectId)}/script-generation/latest`
    ),
    {
      signal,
    },
  );

  if (!response.ok) {
    throw new Error(
      await readErrorMessage(
        response,
        "读取最新剧本生成任务失败。",
      ),
    );
  }

  return response.json();
}


export async function getScriptGenerationScenes(
  runId,
  { signal } = {},
) {
  const response = await fetch(
    (
      `${API_BASE_URL}/api/script-generation/`
      + `${encodeURIComponent(runId)}/scenes`
    ),
    {
      signal,
    },
  );

  if (!response.ok) {
    throw new Error(
      await readErrorMessage(
        response,
        "读取剧本场景失败。",
      ),
    );
  }

  return response.json();
}


export async function updateScriptScene(
  sceneId,
  {
    heading,
    interiorExterior,
    location,
    timeOfDay,
    scriptText,
    characters,
    warnings,
    signal,
  },
) {
  const response = await fetch(
    `${API_BASE_URL}/api/script-scenes/${encodeURIComponent(sceneId)}`,
    {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        heading,
        interior_exterior: interiorExterior,
        location,
        time_of_day: timeOfDay,
        script_text: scriptText,
        characters,
        warnings,
      }),
      signal,
    },
  );

  if (!response.ok) {
    throw new Error(
      await readErrorMessage(
        response,
        "保存剧本场景失败。",
      ),
    );
  }

  return response.json();
}


export async function regenerateScriptScene(
  sceneId,
  {
    instruction = "",
    signal,
  } = {},
) {
  const response = await fetch(
    (
      `${API_BASE_URL}/api/script-scenes/`
      + `${encodeURIComponent(sceneId)}/regenerate`
    ),
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        instruction,
      }),
      signal,
    },
  );

  if (!response.ok) {
    throw new Error(
      await readErrorMessage(
        response,
        "重新生成剧本场景失败。",
      ),
    );
  }

  return response.json();
}
