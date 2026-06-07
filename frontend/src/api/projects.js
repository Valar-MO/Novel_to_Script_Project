const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL
).replace(/\/+$/, "");


export class ApiError extends Error {
  constructor(message, options = {}) {
    super(message);

    this.name = "ApiError";
    this.status = options.status ?? null;
    this.detail = options.detail ?? null;
  }
}


async function readResponseBody(response) {
  const contentType = response.headers.get("content-type") || "";

  if (response.status === 204) {
    return null;
  }

  if (contentType.includes("application/json")) {
    return response.json();
  }

  return response.text();
}


function extractErrorMessage(responseBody, fallbackMessage) {
  if (
    responseBody
    && typeof responseBody === "object"
    && typeof responseBody.detail === "string"
  ) {
    return responseBody.detail;
  }

  if (
    responseBody
    && typeof responseBody === "object"
    && Array.isArray(responseBody.detail)
  ) {
    return responseBody.detail
      .map((item) => item.msg || "请求参数有误")
      .join("；");
  }

  if (typeof responseBody === "string" && responseBody.trim()) {
    return responseBody.trim();
  }

  return fallbackMessage;
}


async function requestJson(
  path,
  {
    method = "GET",
    body = undefined,
    headers = undefined,
    signal = undefined,
  } = {},
) {
  let response;

  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method,
      body,
      headers,
      signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw error;
    }

    throw new ApiError(
      "无法连接后端服务，请确认后端已经启动。",
    );
  }

  const responseBody = await readResponseBody(response);

  if (!response.ok) {
    const message = extractErrorMessage(
      responseBody,
      `请求失败，状态码：${response.status}`,
    );

    throw new ApiError(message, {
      status: response.status,
      detail: responseBody,
    });
  }

  return responseBody;
}


export function uploadProject(
  projectName,
  files,
  { signal } = {},
) {
  const formData = new FormData();

  formData.append("project_name", projectName.trim());

  files.forEach((file) => {
    formData.append("files", file, file.name);
  });

  return requestJson("/api/projects/upload", {
    method: "POST",
    body: formData,
    signal,
  });
}


export function appendProjectFiles(
  projectId,
  files,
  { signal } = {},
) {
  const formData = new FormData();

  files.forEach((file) => {
    formData.append("files", file, file.name);
  });

  return requestJson(
    `/api/projects/${encodeURIComponent(projectId)}/files`,
    {
      method: "POST",
      body: formData,
      signal,
    },
  );
}


export function getProjects(
  { signal } = {},
) {
  return requestJson(
    "/api/projects",
    {
      signal,
    },
  );
}


export function getProjectSummary(
  projectId,
  { signal } = {},
) {
  return requestJson(
    `/api/projects/${encodeURIComponent(projectId)}`,
    {
      signal,
    },
  );
}


export function getProjectChapter(
  projectId,
  chapterId,
  { signal } = {},
) {
  return requestJson(
    (
      `/api/projects/${encodeURIComponent(projectId)}`
      + `/chapters/${encodeURIComponent(chapterId)}`
    ),
    {
      signal,
    },
  );
}


export function getProjectChunk(
  projectId,
  chunkId,
  { signal } = {},
) {
  return requestJson(
    (
      `/api/projects/${encodeURIComponent(projectId)}`
      + `/chunks/${encodeURIComponent(chunkId)}`
    ),
    {
      signal,
    },
  );
}
