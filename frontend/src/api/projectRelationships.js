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


async function requestJson(path, options = {}, fallbackMessage = "请求失败") {
  const response = await fetch(`${API_BASE_URL}${path}`, options);

  if (response.status === 204) {
    return null;
  }

  if (!response.ok) {
    throw new Error(
      await readErrorMessage(response, fallbackMessage),
    );
  }

  return response.json();
}


export function getProjectRelationships(projectId, { signal } = {}) {
  return requestJson(
    `/api/projects/${encodeURIComponent(projectId)}/relationships`,
    { signal },
    "读取人物关系失败。",
  );
}


export function buildProjectRelationships(
  projectId,
  characterRunId,
  { signal } = {},
) {
  return requestJson(
    `/api/projects/${encodeURIComponent(projectId)}/relationships/build`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        character_run_id: characterRunId,
      }),
      signal,
    },
    "构建人物关系失败。",
  );
}


export function createProjectRelationship(projectId, payload) {
  return requestJson(
    `/api/projects/${encodeURIComponent(projectId)}/relationships`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    },
    "新增人物关系失败。",
  );
}


export function updateProjectRelationship(relationshipId, payload) {
  return requestJson(
    `/api/project-relationships/${encodeURIComponent(relationshipId)}`,
    {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    },
    "保存人物关系失败。",
  );
}


export function deleteProjectRelationship(relationshipId) {
  return requestJson(
    `/api/project-relationships/${encodeURIComponent(relationshipId)}`,
    {
      method: "DELETE",
    },
    "删除人物关系失败。",
  );
}

