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


async function requestJson(
  path,
  options = {},
  fallbackMessage = "请求失败。",
) {
  const response = await fetch(
    `${API_BASE_URL}${path}`,
    options,
  );

  if (response.status === 204) {
    return null;
  }

  if (!response.ok) {
    throw new Error(
      await readErrorMessage(
        response,
        fallbackMessage,
      ),
    );
  }

  return response.json();
}


export function buildProjectCharacters(
  projectId,
  narrativeRunId,
  { signal } = {},
) {
  return requestJson(
    `/api/projects/${encodeURIComponent(projectId)}/characters/build`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        narrative_run_id: narrativeRunId,
      }),
      signal,
    },
    "生成人物表失败。",
  );
}


export function getLatestProjectCharacters(
  projectId,
  { signal } = {},
) {
  return requestJson(
    `/api/projects/${encodeURIComponent(projectId)}/characters/latest`,
    {
      signal,
    },
    "读取项目人物失败。",
  );
}


export function deleteProjectCharacter(
  characterRowId,
  { signal } = {},
) {
  return requestJson(
    `/api/project-characters/${encodeURIComponent(
      characterRowId,
    )}`,
    {
      method: "DELETE",
      signal,
    },
    "删除普通人物失败。",
  );
}


export function updateProjectCharacterPin(
  characterRowId,
  isUserPinned,
) {
  return requestJson(
    `/api/project-characters/${encodeURIComponent(characterRowId)}/pin`,
    {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        is_user_pinned: isUserPinned,
      }),
    },
    "更新核心人物失败。",
  );
}
