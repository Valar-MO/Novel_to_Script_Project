import {
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  useNavigate,
  useParams,
} from "react-router-dom";

import {
  ApiError,
  getProjectChunk,
  getProjectChapter,
  getProjectSummary,
} from "../api/projects";
import {
  getActiveNarrativeAnalysisRun,
  getNarrativeAnalysisRun,
  resumeNarrativeAnalysis,
  retryFailedNarrativeAnalysis,
  startNarrativeAnalysis,
} from "../api/narrativeAnalysis";
import {
  getLatestScriptGenerationRun,
  getScriptGenerationRun,
  getScriptGenerationScenes,
  regenerateScriptScene,
  startScriptGeneration,
  updateScriptScene,
} from "../api/scriptGeneration";

import "./ProjectWorkspacePage.css";


const ANALYSIS_TERMINAL_STATUSES = new Set([
  "completed",
  "partial",
  "failed",
  "interrupted",
]);

const ANALYSIS_ACTIVE_STATUSES = new Set([
  "queued",
  "running",
]);

const SCRIPT_TERMINAL_STATUSES = new Set([
  "completed",
  "partial",
  "failed",
  "interrupted",
]);

const SCRIPT_ACTIVE_STATUSES = new Set([
  "queued",
  "running",
]);


function formatBytes(sizeBytes) {
  if (sizeBytes < 1024) {
    return `${sizeBytes} B`;
  }

  if (sizeBytes < 1024 * 1024) {
    return `${(sizeBytes / 1024).toFixed(1)} KB`;
  }

  return `${(
    sizeBytes / (1024 * 1024)
  ).toFixed(2)} MB`;
}


function formatDate(dateText) {
  if (!dateText) {
    return "未知";
  }

  const hasTimezone = (
    dateText.endsWith("Z")
    || /[+-]\d{2}:\d{2}$/.test(dateText)
  );

  const normalizedDateText = hasTimezone
    ? dateText
    : `${dateText.replace(" ", "T")}Z`;

  const parsedDate = new Date(normalizedDateText);

  if (Number.isNaN(parsedDate.getTime())) {
    return dateText;
  }

  return parsedDate.toLocaleString("zh-CN", {
    hour12: false,
  });
}


function getStatusLabel(status) {
  const statusLabels = {
    created: "项目已创建",
    preprocessed: "文本预处理完成",
    processing: "正在处理",
    failed: "处理失败",
  };

  return statusLabels[status] || status;
}


function getChapterDisplayName(chapter) {
  if (chapter.full_title) {
    return chapter.full_title;
  }

  if (chapter.chapter_title) {
    return chapter.chapter_title;
  }

  return chapter.is_detected
    ? `第 ${chapter.chapter_order} 章`
    : "未识别章节";
}


function findFirstChapter(project) {
  for (const file of project?.files || []) {
    if (file.chapters?.length > 0) {
      return file.chapters[0];
    }
  }

  return null;
}


function getErrorMessage(error, fallbackMessage) {
  if (error instanceof ApiError) {
    return error.message;
  }

  if (error instanceof Error) {
    return error.message;
  }

  return fallbackMessage;
}


function getMentionTypeLabel(mentionType) {
  const labels = {
    character: "人物",
    location: "地点",
    time: "时间",
    organization: "组织",
    object: "物件",
  };

  return labels[mentionType] || mentionType;
}


function getEventTypeLabel(eventType) {
  const labels = {
    movement: "移动",
    communication: "交流",
    perception: "感知",
    cognition: "认知",
    state: "状态",
    possession: "持有",
    social: "社会行为",
    creation: "创建",
    conflict: "冲突",
    other: "其他",
  };

  return labels[eventType] || eventType;
}


function ProjectWorkspacePage() {
  const { projectId } = useParams();
  const navigate = useNavigate();

  const chapterCacheRef = useRef(new Map());
  const readerTextRef = useRef(null);
  const highlightedEvidenceRef = useRef(null);

  const [project, setProject] = useState(null);
  const [projectLoading, setProjectLoading] = useState(true);
  const [projectError, setProjectError] = useState("");
  const [projectNotFound, setProjectNotFound] = useState(false);

  const [expandedFileIds, setExpandedFileIds] = useState(
    new Set(),
  );

  const [selectedChapterId, setSelectedChapterId] = useState(
    null,
  );
  const [selectedChapter, setSelectedChapter] = useState(null);
  const [chapterLoading, setChapterLoading] = useState(false);
  const [chapterError, setChapterError] = useState("");
  const [chapterReloadKey, setChapterReloadKey] = useState(0);

  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [analysisError, setAnalysisError] = useState("");
  const [activeMention, setActiveMention] = useState(null);
  const [
    showMentionDebug,
    setShowMentionDebug,
  ] = useState(false);
  const [isGeneratingScript, setIsGeneratingScript] = useState(false);
  const [scriptRun, setScriptRun] = useState(null);
  const [scriptScenes, setScriptScenes] = useState([]);
  const [scriptError, setScriptError] = useState("");
  const [selectedScriptSceneId, setSelectedScriptSceneId] = useState(null);
  const [editingSceneId, setEditingSceneId] = useState(null);
  const [sceneDraft, setSceneDraft] = useState({
    heading: "",
    scriptText: "",
  });
  const [regeneratingSceneId, setRegeneratingSceneId] = useState(null);
  const [regenerateInstruction, setRegenerateInstruction] = useState("");
  const [scriptModel, setScriptModel] = useState("deepseek-v4-pro");
  const [scriptThinkingEnabled, setScriptThinkingEnabled] = useState(false);

  const selectedChapterSummary = useMemo(() => {
    for (const file of project?.files || []) {
      const chapter = file.chapters?.find(
        (item) => item.id === selectedChapterId,
      );

      if (chapter) {
        return chapter;
      }
    }

    return null;
  }, [project, selectedChapterId]);

  useEffect(() => {
    const controller = new AbortController();

    setProject(null);
    setProjectLoading(true);
    setProjectError("");
    setProjectNotFound(false);

    setSelectedChapterId(null);
    setSelectedChapter(null);
    setChapterError("");
    setChapterReloadKey(0);

    chapterCacheRef.current.clear();

    getProjectSummary(projectId, {
      signal: controller.signal,
    })
      .then((projectData) => {
        setProject(projectData);

        setExpandedFileIds(
          new Set(
            projectData.files.map((file) => file.id),
          ),
        );

        const firstChapter = findFirstChapter(projectData);

        if (firstChapter) {
          setSelectedChapterId(firstChapter.id);
        }
      })
      .catch((error) => {
        if (
          error instanceof DOMException
          && error.name === "AbortError"
        ) {
          return;
        }

        if (
          error instanceof ApiError
          && error.status === 404
        ) {
          setProjectNotFound(true);
          setProjectError(
            "该项目不存在或已经被删除。",
          );
          return;
        }

        setProjectError(
          getErrorMessage(
            error,
            "项目加载失败，请稍后重试。",
          ),
        );
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setProjectLoading(false);
        }
      });

    return () => {
      controller.abort();
    };
  }, [projectId]);

  useEffect(() => {
    if (!selectedChapterId) {
      setSelectedChapter(null);
      setChapterLoading(false);
      setChapterError("");
      return undefined;
    }

    const cacheKey = String(selectedChapterId);
    const cachedChapter = chapterCacheRef.current.get(
      cacheKey,
    );

    if (cachedChapter && chapterReloadKey === 0) {
      setSelectedChapter(cachedChapter);
      setChapterLoading(false);
      setChapterError("");
      return undefined;
    }

    const controller = new AbortController();

    setSelectedChapter(null);
    setChapterLoading(true);
    setChapterError("");

    getProjectChapter(
      projectId,
      selectedChapterId,
      {
        signal: controller.signal,
      },
    )
      .then((chapterData) => {
        chapterCacheRef.current.set(
          cacheKey,
          chapterData,
        );

        setSelectedChapter(chapterData);
      })
      .catch((error) => {
        if (
          error instanceof DOMException
          && error.name === "AbortError"
        ) {
          return;
        }

        setChapterError(
          getErrorMessage(
            error,
            "章节正文加载失败，请稍后重试。",
          ),
        );
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setChapterLoading(false);
        }
      });

    return () => {
      controller.abort();
    };
  }, [
    projectId,
    selectedChapterId,
    chapterReloadKey,
  ]);

  useEffect(() => {
    if (projectLoading || projectError) {
      return undefined;
    }

    const controller = new AbortController();

    getActiveNarrativeAnalysisRun(projectId, {
      signal: controller.signal,
    })
      .then((activeRun) => {
        if (activeRun) {
          setAnalysisResult(activeRun);
        }
      })
      .catch((error) => {
        if (
          error instanceof DOMException
          && error.name === "AbortError"
        ) {
          return;
        }

        setAnalysisError(
          getErrorMessage(
            error,
            "读取当前 AI 分析任务失败。",
          ),
        );
      });

    return () => {
      controller.abort();
    };
  }, [
    projectId,
    projectLoading,
    projectError,
  ]);

  useEffect(() => {
    if (projectLoading || projectError) {
      return undefined;
    }

    const controller = new AbortController();

    getLatestScriptGenerationRun(projectId, {
      signal: controller.signal,
    })
      .then(async (latestRun) => {
        if (!latestRun) {
          return;
        }

        setScriptRun(latestRun);

        const scenesData = await getScriptGenerationScenes(
          latestRun.id,
          {
            signal: controller.signal,
          },
        );
        setScriptScenes(scenesData.scenes || []);
        setSelectedScriptSceneId((currentId) => (
          currentId || scenesData.scenes?.[0]?.id || null
        ));
      })
      .catch((error) => {
        if (
          error instanceof DOMException
          && error.name === "AbortError"
        ) {
          return;
        }

        setScriptError(
          getErrorMessage(
            error,
            "读取最新剧本生成结果失败。",
          ),
        );
      });

    return () => {
      controller.abort();
    };
  }, [
    projectId,
    projectLoading,
    projectError,
  ]);

  useEffect(() => {
    const runId = analysisResult?.id || analysisResult?.run_id;

    if (!runId || !ANALYSIS_ACTIVE_STATUSES.has(analysisResult.status)) {
      return undefined;
    }

    let timeoutId = null;
    const controller = new AbortController();

    async function pollRun() {
      try {
        const runDetail = await getNarrativeAnalysisRun(
          runId,
          {
            includeUnits: false,
            signal: controller.signal,
          },
        );

        if (ANALYSIS_TERMINAL_STATUSES.has(runDetail.status)) {
          const runWithUnits = await getNarrativeAnalysisRun(
            runId,
            {
              includeUnits: true,
              signal: controller.signal,
            },
          );
          setAnalysisResult(runWithUnits);
          setIsAnalyzing(false);
          return;
        }

        setAnalysisResult((current) => ({
          ...(current || {}),
          ...runDetail,
          units: current?.units || [],
        }));
        setIsAnalyzing(true);
        timeoutId = window.setTimeout(pollRun, 2500);
      } catch (error) {
        if (
          error instanceof DOMException
          && error.name === "AbortError"
        ) {
          return;
        }

        setIsAnalyzing(false);
        setAnalysisError(
          getErrorMessage(
            error,
            "刷新 AI 分析进度失败。",
          ),
        );
      }
    }

    timeoutId = window.setTimeout(pollRun, 800);

    return () => {
      controller.abort();

      if (timeoutId !== null) {
        window.clearTimeout(timeoutId);
      }
    };
  }, [
    analysisResult?.id,
    analysisResult?.run_id,
    analysisResult?.status,
  ]);

  useEffect(() => {
    const runId = scriptRun?.id || scriptRun?.run_id;

    if (scriptRun?.status && SCRIPT_TERMINAL_STATUSES.has(scriptRun.status)) {
      setIsGeneratingScript(false);
    }

    if (!runId || !SCRIPT_ACTIVE_STATUSES.has(scriptRun.status)) {
      return undefined;
    }

    let timeoutId = null;
    const controller = new AbortController();

    async function pollScriptRun() {
      try {
        const runDetail = await getScriptGenerationRun(
          runId,
          {
            includeUnits: false,
            signal: controller.signal,
          },
        );

        setScriptRun((current) => ({
          ...(current || {}),
          ...runDetail,
          units: current?.units || [],
        }));

        const scenesData = await getScriptGenerationScenes(
          runId,
          {
            signal: controller.signal,
          },
        );
        setScriptScenes(scenesData.scenes || []);
        setSelectedScriptSceneId((currentId) => (
          currentId || scenesData.scenes?.[0]?.id || null
        ));

        if (SCRIPT_TERMINAL_STATUSES.has(runDetail.status)) {
          const runWithUnits = await getScriptGenerationRun(
            runId,
            {
              includeUnits: true,
              signal: controller.signal,
            },
          );
          setScriptRun(runWithUnits);
          setIsGeneratingScript(false);
          return;
        }

        setIsGeneratingScript(true);
        timeoutId = window.setTimeout(pollScriptRun, 2500);
      } catch (error) {
        if (
          error instanceof DOMException
          && error.name === "AbortError"
        ) {
          return;
        }

        setIsGeneratingScript(false);
        setScriptError(
          getErrorMessage(
            error,
            "刷新剧本生成进度失败。",
          ),
        );
      }
    }

    timeoutId = window.setTimeout(pollScriptRun, 800);

    return () => {
      controller.abort();

      if (timeoutId !== null) {
        window.clearTimeout(timeoutId);
      }
    };
  }, [
    scriptRun?.id,
    scriptRun?.run_id,
    scriptRun?.status,
  ]);

  useEffect(() => {
    if (!highlightedEvidenceRef.current) {
      return;
    }

    highlightedEvidenceRef.current.scrollIntoView({
      behavior: "smooth",
      block: "center",
      inline: "nearest",
    });
  }, [activeMention, selectedChapter]);

  const highlightedChapterText = useMemo(() => {
    const chapterText = selectedChapter?.text || "";
    const evidenceText = activeMention?.evidence_text || "";

    if (!chapterText || !evidenceText) {
      return {
        before: chapterText,
        match: "",
        after: "",
      };
    }

    const startIndex = chapterText.indexOf(evidenceText);

    if (startIndex < 0) {
      return {
        before: chapterText,
        match: "",
        after: "",
      };
    }

    const endIndex = startIndex + evidenceText.length;

    return {
      before: chapterText.slice(0, startIndex),
      match: chapterText.slice(startIndex, endIndex),
      after: chapterText.slice(endIndex),
    };
  }, [activeMention, selectedChapter]);

  const selectedScriptScene = useMemo(() => (
    scriptScenes.find((scene) => scene.id === selectedScriptSceneId)
    || scriptScenes[0]
    || null
  ), [scriptScenes, selectedScriptSceneId]);

  function toggleFile(fileId) {
    setExpandedFileIds((currentSet) => {
      const nextSet = new Set(currentSet);

      if (nextSet.has(fileId)) {
        nextSet.delete(fileId);
      } else {
        nextSet.add(fileId);
      }

      return nextSet;
    });
  }

  function selectChapter(chapterId) {
    setSelectedChapterId(chapterId);
    setChapterReloadKey(0);
  }

  function retryChapterLoad() {
    if (!selectedChapterId) {
      return;
    }

    chapterCacheRef.current.delete(
      String(selectedChapterId),
    );

    setChapterReloadKey(
      (currentKey) => currentKey + 1,
    );
  }

  async function handleStartAnalysis() {
    setIsAnalyzing(true);
    setAnalysisError("");
    setActiveMention(null);
    setShowMentionDebug(false);

    try {
      const result = await startNarrativeAnalysis(
        projectId,
        {
          maxChunks: null,
          forceReanalyze: false,
        },
      );

      setAnalysisResult({
        id: result.run_id,
        ...result,
        units: [],
      });
    } catch (error) {
      setAnalysisError(
        error instanceof Error
          ? error.message
          : "AI analysis failed.",
      );
    }
  }

  async function handleResumeAnalysis() {
    const runId = analysisResult?.id || analysisResult?.run_id;

    if (!runId) {
      return;
    }

    setIsAnalyzing(true);
    setAnalysisError("");

    try {
      const result = await resumeNarrativeAnalysis(runId);

      setAnalysisResult({
        id: result.run_id,
        ...result,
        units: analysisResult?.units || [],
      });
    } catch (error) {
      setIsAnalyzing(false);
      setAnalysisError(
        getErrorMessage(
          error,
          "继续 AI 分析失败。",
        ),
      );
    }
  }

  async function handleRetryFailedAnalysis() {
    const runId = analysisResult?.id || analysisResult?.run_id;

    if (!runId) {
      return;
    }

    setIsAnalyzing(true);
    setAnalysisError("");

    try {
      const result = await retryFailedNarrativeAnalysis(runId);

      setAnalysisResult({
        id: result.run_id,
        ...result,
        units: analysisResult?.units || [],
      });
    } catch (error) {
      setIsAnalyzing(false);
      setAnalysisError(
        getErrorMessage(
          error,
          "重试失败文本块失败。",
        ),
      );
    }
  }

  async function refreshScriptScenes(runId) {
    const scenesData = await getScriptGenerationScenes(runId);
    setScriptScenes(scenesData.scenes || []);
    setSelectedScriptSceneId((currentId) => (
      currentId || scenesData.scenes?.[0]?.id || null
    ));
  }

  async function handleStartScriptGeneration() {
    setIsGeneratingScript(true);
    setScriptError("");
    setScriptScenes([]);
    setSelectedScriptSceneId(null);
    setEditingSceneId(null);

    try {
      const result = await startScriptGeneration(
        projectId,
        {
          maxChunks: null,
          generationStyle: "standard",
          adaptationMode: "faithful",
          provider: "deepseek",
          model: scriptModel,
          thinkingEnabled: scriptThinkingEnabled,
        },
      );

      setScriptRun({
        id: result.run_id,
        ...result,
        units: [],
      });
    } catch (error) {
      setIsGeneratingScript(false);
      setScriptError(
        getErrorMessage(
          error,
          "启动剧本生成失败。",
        ),
      );
    }
  }

  async function handleSelectScriptScene(scene) {
    setSelectedScriptSceneId(scene.id);

    const firstSource = scene.source_spans?.[0];
    if (!firstSource?.evidence_text) {
      return;
    }

    setActiveMention({
      evidence_text: firstSource.evidence_text,
    });

    if (firstSource.chunk_id) {
      try {
        const chunk = await getProjectChunk(
          projectId,
          firstSource.chunk_id,
        );

        if (chunk?.chapter_id && chunk.chapter_id !== selectedChapterId) {
          setSelectedChapterId(chunk.chapter_id);
          setChapterReloadKey(0);
        }
      } catch (error) {
        setScriptError(
          getErrorMessage(
            error,
            "读取场景对应原文失败。",
          ),
        );
      }
    }
  }

  function handleHighlightSceneSource(scene) {
    const firstSource = scene.source_spans?.[0];
    if (firstSource?.evidence_text) {
      setActiveMention({
        evidence_text: firstSource.evidence_text,
      });
    }
  }

  function handleStartSceneEdit(scene) {
    setEditingSceneId(scene.id);
    setSceneDraft({
      heading: scene.heading,
      interiorExterior: scene.interior_exterior,
      location: scene.location,
      timeOfDay: scene.time_of_day,
      scriptText: scene.script_text,
    });
  }

  function handleCancelSceneEdit() {
    setEditingSceneId(null);
    setSceneDraft({
      heading: "",
      interiorExterior: "",
      location: "",
      timeOfDay: "",
      scriptText: "",
    });
  }

  async function handleSaveScene(scene) {
    setScriptError("");

    try {
      const updatedScene = await updateScriptScene(
        scene.id,
        {
          heading: sceneDraft.heading,
          interiorExterior: sceneDraft.interiorExterior,
          location: sceneDraft.location,
          timeOfDay: sceneDraft.timeOfDay,
          scriptText: sceneDraft.scriptText,
          characters: scene.characters,
          warnings: scene.warnings,
        },
      );

      setScriptScenes((currentScenes) => (
        currentScenes.map((item) => (
          item.id === updatedScene.id ? updatedScene : item
        ))
      ));
      setSelectedScriptSceneId(updatedScene.id);
      handleCancelSceneEdit();
    } catch (error) {
      setScriptError(
        getErrorMessage(
          error,
          "保存剧本场景失败。",
        ),
      );
    }
  }

  async function handleRegenerateScene(scene) {
    setScriptError("");
    setRegeneratingSceneId(scene.id);

    try {
      const regeneratedScene = await regenerateScriptScene(
        scene.id,
        {
          instruction: regenerateInstruction,
        },
      );

      setScriptScenes((currentScenes) => (
        currentScenes.map((item) => (
          item.id === regeneratedScene.id ? regeneratedScene : item
        ))
      ));
      setSelectedScriptSceneId(regeneratedScene.id);
      setRegenerateInstruction("");
    } catch (error) {
      setScriptError(
        getErrorMessage(
          error,
          "重新生成剧本场景失败。",
        ),
      );
    } finally {
      setRegeneratingSceneId(null);
    }
  }

  if (projectLoading) {
    return (
      <main className="project-workspace-state-page">
        <div className="project-workspace-loader" />
        <h1>正在加载项目</h1>
        <p>正在从数据库读取项目摘要...</p>
      </main>
    );
  }

  if (projectError) {
    return (
      <main className="project-workspace-state-page">
        <span className="project-workspace-state-icon">
          {projectNotFound ? "404" : "!"}
        </span>

        <h1>
          {projectNotFound
            ? "Project not found"
            : "项目加载失败"}
        </h1>

        <p>{projectError}</p>

        <div className="project-workspace-state-actions">
          <button
            type="button"
            onClick={() => navigate("/projects")}
          >
            返回项目列表
          </button>

          <button
            type="button"
            className="primary"
            onClick={() => navigate("/project/new")}
          >
            创建新项目
          </button>
        </div>
      </main>
    );
  }

  const readerTitle = selectedChapter
    ? getChapterDisplayName(selectedChapter)
    : selectedChapterSummary
      ? getChapterDisplayName(selectedChapterSummary)
      : "尚未选择章节";

  function handleMentionClick(mention) {
    setActiveMention(mention);
  }

  return (
    <main className="project-workspace-page">
      <header className="project-workspace-header">
        <div className="project-workspace-navigation">
          <button
            type="button"
            onClick={() => navigate("/projects")}
          >
            ← 项目列表
          </button>

          <button
            type="button"
            onClick={() => navigate("/")}
          >
            首页
          </button>

          <button
            type="button"
            onClick={() => navigate("/project/new")}
          >
            + 新建项目
          </button>
        </div>

        <div className="project-workspace-title-row">
          <div>
            <p>Novel2Script 项目工作区</p>
            <h1>{project.project_name}</h1>
          </div>

          <span className="project-workspace-status">
            {getStatusLabel(project.status)}
          </span>
        </div>

        <div className="project-workspace-statistics">
          <div>
            <span>源文件</span>
            <strong>{project.file_count}</strong>
          </div>

          <div>
            <span>章节</span>
            <strong>{project.chapter_count}</strong>
          </div>

          <div>
            <span>创建时间</span>
            <strong>
              {formatDate(project.created_at)}
            </strong>
          </div>
        </div>

        <section className="project-workspace-analysis">
          <div className="project-workspace-analysis-heading">
            <div>
              <strong>AI 叙事分析</strong>
              <span>按当前项目的全部文本块串行分析</span>
            </div>

            <button
              type="button"
              className="project-workspace-analysis-button"
              disabled={isAnalyzing}
              onClick={handleStartAnalysis}
            >
              {isAnalyzing
                ? "Analysis running..."
                : "开始 AI 分析"}
            </button>

            {analysisResult?.status === "interrupted" && (
              <button
                type="button"
                className="project-workspace-analysis-button"
                disabled={isAnalyzing}
                onClick={handleResumeAnalysis}
              >
                继续分析
              </button>
            )}

            {(analysisResult?.failed_chunks ?? 0) > 0 && (
              <button
                type="button"
                className="project-workspace-analysis-button"
                disabled={isAnalyzing}
                onClick={handleRetryFailedAnalysis}
              >
                重试失败文本块
              </button>
            )}
          </div>

          {analysisError && (
            <p className="project-workspace-analysis-error">
              {analysisError}
            </p>
          )}

          {analysisResult && (
            <div className="project-workspace-analysis-summary">
              <div>
                <span>状态</span>
                <strong>{analysisResult.status}</strong>
              </div>

              <div>
                <span>批次</span>
                <strong>{analysisResult.id}</strong>
              </div>
              <div>
                <span>进度</span>
                <strong>
                  {analysisResult.processed_chunks ?? 0}
                  {" / "}
                  {analysisResult.total_chunks ?? 0}
                </strong>
              </div>

              <div>
                <span>成功</span>
                <strong>{analysisResult.successful_chunks ?? 0}</strong>
              </div>

              <div>
                <span>部分成功</span>
                <strong>{analysisResult.partial_chunks ?? 0}</strong>
              </div>

              <div>
                <span>失败</span>
                <strong>{analysisResult.failed_chunks ?? 0}</strong>
              </div>

              <div>
                <span>缓存层</span>
                <strong>{analysisResult.cached_layers ?? 0}</strong>
              </div>

              <div>
                <span>缓存块</span>
                <strong>{analysisResult.cached_chunks ?? 0}</strong>
              </div>

              <div>
                <span>当前文本块</span>
                <strong>{analysisResult.current_chunk_id || "-"}</strong>
              </div>
            </div>
          )}

          {analysisResult?.units?.length > 0 && (
            <div className="project-workspace-mentions">
              {analysisResult.units.map((unit) => {
                const mentions = (
                  unit.validated_result?.mentions
                  || unit.result?.mentions
                  || []
                );
                const relations = (
                  unit.validated_result?.relations
                  || unit.result?.relations
                  || []
                );
                const eventFrames = (
                  unit.validated_result?.event_frames
                  || unit.result?.event_frames
                  || []
                );
                const characterCandidates = (
                  unit.validated_result?.character_candidates
                  || unit.result?.character_candidates
                  || []
                );

                return (
                  <section
                    key={unit.id}
                    className="project-workspace-mention-unit"
                  >
                    <div className="project-workspace-mention-unit-heading">
                      <strong>{unit.chunk_id}</strong>
                      <span>{unit.status}</span>
                    </div>

                    {unit.error_message && (
                      <p className="project-workspace-analysis-error">
                        {unit.error_message}
                      </p>
                    )}

                    {unit.validated_result?.layer_statuses && (
                      <div className="project-workspace-layer-statuses">
                        {Object.entries(
                          unit.validated_result.layer_statuses,
                        ).map(([layerName, layerStatus]) => (
                          <span key={layerName}>
                            {layerName}
                            {": "}
                            {layerStatus}
                          </span>
                        ))}
                      </div>
                    )}

                    {characterCandidates.length > 0 && (
                      <>
                        <h4 className="project-workspace-result-title">
                          人物候选
                        </h4>
                        <div className="project-workspace-result-list">
                          {characterCandidates.map((candidate, index) => (
                            <button
                              type="button"
                              key={
                                candidate.character_candidate_id
                                || `${candidate.canonical_name}-${index}`
                              }
                              className={
                                activeMention === candidate
                                  ? "project-workspace-result-item active"
                                  : "project-workspace-result-item"
                              }
                              onClick={() => {
                                handleMentionClick(candidate);
                              }}
                            >
                              <strong>{candidate.canonical_name}</strong>
                              <span>
                                {
                                  (
                                    (candidate.aliases || []).length > 0
                                      ? candidate.aliases
                                      : candidate.references || []
                                  ).join(" / ")
                                }
                              </span>
                              {(candidate.references || []).length > 0 && (
                                <small>
                                  references:
                                  {" "}
                                  {(candidate.references || []).join(" / ")}
                                </small>
                              )}
                              <small>
                                置信度
                                {" "}
                                {Number(
                                  candidate.confidence,
                                ).toFixed(2)}
                              </small>
                            </button>
                          ))}
                        </div>
                      </>
                    )}

                    {relations.length > 0 && (
                      <>
                        <h4 className="project-workspace-result-title">
                          关系
                        </h4>
                        <div className="project-workspace-result-list">
                          {relations.map((relation, index) => (
                            <button
                              type="button"
                              key={
                                relation.relation_id
                                || `${relation.source_mention}-${relation.target_mention}-${index}`
                              }
                              className={
                                activeMention === relation
                                  ? "project-workspace-result-item active"
                                  : "project-workspace-result-item"
                              }
                              onClick={() => {
                                handleMentionClick(relation);
                              }}
                            >
                              <strong>
                                {relation.source_mention}
                                {" → "}
                                {relation.target_mention}
                              </strong>
                              <span>{relation.relation}</span>
                              <small>
                                置信度
                                {" "}
                                {Number(
                                  relation.confidence,
                                ).toFixed(2)}
                              </small>
                            </button>
                          ))}
                        </div>
                      </>
                    )}

                    {eventFrames.length > 0 && (
                      <>
                        <h4 className="project-workspace-result-title">
                          事件
                        </h4>
                        <div className="project-workspace-result-list">
                          {eventFrames.map((eventFrame, index) => (
                            <button
                              type="button"
                              key={
                                eventFrame.event_frame_id
                                || `${eventFrame.trigger_text}-${index}`
                              }
                              className={
                                activeMention === eventFrame
                                  ? "project-workspace-result-item active"
                                  : "project-workspace-result-item"
                              }
                              onClick={() => {
                                handleMentionClick(eventFrame);
                              }}
                            >
                              <strong>{eventFrame.trigger_text}</strong>
                              <span>
                                {getEventTypeLabel(
                                  eventFrame.event_type,
                                )}
                              </span>
                              <small>
                                {(eventFrame.arguments || [])
                                  .map((argument) => (
                                    `${argument.role}: ${argument.mention_text}`
                                  ))
                                  .join(" · ")}
                              </small>
                            </button>
                          ))}
                        </div>
                      </>
                    )}

                    <div className="project-workspace-debug-heading">
                      <h4 className="project-workspace-result-title">
                        文本锚点
                      </h4>
                      <button
                        type="button"
                        onClick={() => {
                          setShowMentionDebug(
                            (currentValue) => !currentValue,
                          );
                        }}
                      >
                        {showMentionDebug ? "隐藏" : "展开"}
                      </button>
                    </div>

                    {showMentionDebug && (
                      mentions.length === 0 ? (
                        <p className="project-workspace-mention-empty">
                          未识别到文本锚点
                        </p>
                      ) : (
                        <div className="project-workspace-mention-list">
                          {mentions.map((mention, index) => (
                            <button
                              type="button"
                              key={`${mention.mention_text}-${index}`}
                              className={
                                activeMention === mention
                                  ? "project-workspace-mention-item active"
                                  : "project-workspace-mention-item"
                              }
                              onClick={() => {
                                handleMentionClick(mention);
                              }}
                            >
                              <span>
                                {getMentionTypeLabel(
                                  mention.mention_type,
                                )}
                              </span>
                              <strong>{mention.mention_text}</strong>
                              <small>
                                {mention.evidence_validated === false
                                  ? "Evidence not located"
                                  : "Evidence located"}
                                {" · "}
                                置信度
                                {" "}
                                {Number(
                                  mention.confidence,
                                ).toFixed(2)}
                              </small>
                            </button>
                          ))}
                        </div>
                      )
                    )}
                  </section>
                );
              })}
            </div>
          )}
        </section>

        <section className="project-workspace-script">
          <div className="project-workspace-analysis-heading">
            <div>
              <strong>剧本场景生成</strong>
              <span>直接根据小说文本块生成可编辑剧本场景</span>
            </div>

            <button
              type="button"
              className="project-workspace-analysis-button"
              disabled={isGeneratingScript}
              onClick={handleStartScriptGeneration}
            >
              {isGeneratingScript ? "正在生成..." : "生成剧本"}
            </button>
          </div>

          {scriptError && (
            <p className="project-workspace-analysis-error">
              {scriptError}
            </p>
          )}

          <div className="project-workspace-script-controls">
            <label>
              <span>模型</span>
              <select
                value={scriptModel}
                disabled={isGeneratingScript}
                onChange={(event) => {
                  setScriptModel(event.target.value);
                }}
              >
                <option value="deepseek-v4-pro">
                  DeepSeek V4 Pro
                </option>
              </select>
            </label>

            <label className="project-workspace-script-switch">
              <input
                type="checkbox"
                checked={scriptThinkingEnabled}
                disabled={isGeneratingScript}
                onChange={(event) => {
                  setScriptThinkingEnabled(event.target.checked);
                }}
              />
              <span>深度思考</span>
            </label>
          </div>

          {scriptRun && (
            <div className="project-workspace-analysis-summary">
              <div>
                <span>状态</span>
                <strong>{scriptRun.status}</strong>
              </div>

              <div>
                <span>批次</span>
                <strong>{scriptRun.id}</strong>
              </div>

              <div>
                <span>模型</span>
                <strong>
                  {scriptRun.provider || "-"}
                  {scriptRun.model ? ` / ${scriptRun.model}` : ""}
                </strong>
              </div>

              <div>
                <span>进度</span>
                <strong>
                  {scriptRun.processed_chunks ?? 0}
                  {" / "}
                  {scriptRun.total_chunks ?? 0}
                </strong>
              </div>

              <div>
                <span>场景</span>
                <strong>{scriptRun.scene_count ?? scriptScenes.length}</strong>
              </div>

              <div>
                <span>部分成功</span>
                <strong>{scriptRun.partial_chunks ?? 0}</strong>
              </div>

              <div>
                <span>失败</span>
                <strong>{scriptRun.failed_chunks ?? 0}</strong>
              </div>
            </div>
          )}

          {scriptScenes.length > 0 && (
            <div className="project-workspace-script-grid">
              <div className="project-workspace-script-list">
                {scriptScenes.map((scene) => (
                  <button
                    type="button"
                    key={scene.id}
                    className={
                      selectedScriptScene?.id === scene.id
                        ? "project-workspace-script-list-item active"
                        : "project-workspace-script-list-item"
                    }
                    onClick={() => handleSelectScriptScene(scene)}
                  >
                    <strong>
                      第
                      {scene.scene_number}
                      场
                    </strong>
                    <span>{scene.heading}</span>
                    <small>
                      {scene.is_user_edited ? "已编辑" : "AI 生成"}
                      {scene.warnings?.length > 0
                        ? ` · ${scene.warnings.length} 条警告`
                        : ""}
                    </small>
                  </button>
                ))}
              </div>

              {selectedScriptScene && (
                <article className="project-workspace-script-detail">
                  {editingSceneId === selectedScriptScene.id ? (
                    <>
                      <label>
                        <span>场景标题</span>
                        <input
                          value={sceneDraft.heading}
                          onChange={(event) => {
                            setSceneDraft((currentDraft) => ({
                              ...currentDraft,
                              heading: event.target.value,
                            }));
                          }}
                        />
                      </label>

                      <div className="project-workspace-scene-meta-editor">
                        <label>
                          <span>景别</span>
                          <input
                            value={sceneDraft.interiorExterior}
                            onChange={(event) => {
                              setSceneDraft((currentDraft) => ({
                                ...currentDraft,
                                interiorExterior: event.target.value,
                              }));
                            }}
                          />
                        </label>

                        <label>
                          <span>地点</span>
                          <input
                            value={sceneDraft.location}
                            onChange={(event) => {
                              setSceneDraft((currentDraft) => ({
                                ...currentDraft,
                                location: event.target.value,
                              }));
                            }}
                          />
                        </label>

                        <label>
                          <span>时间</span>
                          <input
                            value={sceneDraft.timeOfDay}
                            onChange={(event) => {
                              setSceneDraft((currentDraft) => ({
                                ...currentDraft,
                                timeOfDay: event.target.value,
                              }));
                            }}
                          />
                        </label>
                      </div>

                      <label>
                        <span>剧本正文</span>
                        <textarea
                          value={sceneDraft.scriptText}
                          onChange={(event) => {
                            setSceneDraft((currentDraft) => ({
                              ...currentDraft,
                              scriptText: event.target.value,
                            }));
                          }}
                        />
                      </label>

                      <div className="project-workspace-script-actions">
                        <button
                          type="button"
                          onClick={() => handleSaveScene(selectedScriptScene)}
                        >
                          保存
                        </button>
                        <button
                          type="button"
                          className="secondary"
                          onClick={handleCancelSceneEdit}
                        >
                          取消
                        </button>
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="project-workspace-script-detail-heading">
                        <div>
                          <span>
                            第
                            {selectedScriptScene.scene_number}
                            场
                          </span>
                          <h3>{selectedScriptScene.heading}</h3>
                        </div>
                        <div className="project-workspace-script-actions">
                          <button
                            type="button"
                            className="secondary"
                            onClick={() => handleSelectScriptScene(selectedScriptScene)}
                          >
                            查看原文
                          </button>
                          <button
                            type="button"
                            onClick={() => handleStartSceneEdit(selectedScriptScene)}
                          >
                            编辑
                          </button>
                          <button
                            type="button"
                            className="secondary"
                            disabled={regeneratingSceneId === selectedScriptScene.id}
                            onClick={() => handleRegenerateScene(selectedScriptScene)}
                          >
                            {regeneratingSceneId === selectedScriptScene.id
                              ? "生成中"
                              : "重生成"}
                          </button>
                        </div>
                      </div>

                      <div className="project-workspace-script-meta">
                        <span>{selectedScriptScene.interior_exterior}</span>
                        <span>{selectedScriptScene.location}</span>
                        <span>{selectedScriptScene.time_of_day}</span>
                      </div>

                      {selectedScriptScene.characters?.length > 0 && (
                        <p className="project-workspace-script-characters">
                          人物：
                          {selectedScriptScene.characters
                            .map((character) => character.name)
                            .join("、")}
                        </p>
                      )}

                      <pre className="project-workspace-script-text">
                        {selectedScriptScene.script_text}
                      </pre>

                      <label className="project-workspace-regenerate-box">
                        <span>重生成要求</span>
                        <input
                          value={regenerateInstruction}
                          placeholder="例如：更忠实原文，减少镜头术语"
                          onChange={(event) => {
                            setRegenerateInstruction(event.target.value);
                          }}
                        />
                      </label>

                      {selectedScriptScene.warnings?.length > 0 && (
                        <div className="project-workspace-script-warnings">
                          {selectedScriptScene.warnings.map((warning) => (
                            <span key={warning}>{warning}</span>
                          ))}
                        </div>
                      )}
                    </>
                  )}
                </article>
              )}
            </div>
          )}
        </section>
      </header>

      <section className="project-workspace-layout">
        <aside className="project-workspace-sidebar">
          <div className="project-workspace-sidebar-heading">
            <div>
              <h2>小说目录</h2>
              <span>选择章节查看完整原文</span>
            </div>
          </div>

          {project.files.length === 0 ? (
            <div className="project-workspace-empty">
              当前项目没有源文件。
            </div>
          ) : (
            <div className="project-workspace-tree">
              {project.files.map((file) => {
                const fileExpanded = (
                  expandedFileIds.has(file.id)
                );

                return (
                  <section
                    key={file.id}
                    className="project-workspace-file"
                  >
                    <button
                      type="button"
                      className="project-workspace-file-row"
                      onClick={() => toggleFile(file.id)}
                    >
                      <span className="project-workspace-chevron">
                        {fileExpanded ? "v" : ">"}
                      </span>

                      <span className="project-workspace-file-icon">
                        TXT
                      </span>

                      <span className="project-workspace-tree-text">
                        <strong>{file.file_name}</strong>

                        <small>
                          {formatBytes(file.size_bytes)}
                          {" · "}
                          {file.chapter_count} 章
                        </small>
                      </span>
                    </button>

                    {fileExpanded && (
                      <div className="project-workspace-chapter-list">
                        {file.chapters.length === 0 ? (
                          <div className="project-workspace-no-chapters">
                            未检测到章节
                          </div>
                        ) : (
                          file.chapters.map((chapter) => (
                            <button
                              key={chapter.id}
                              type="button"
                              className={
                                chapter.id === selectedChapterId
                                  ? "project-workspace-chapter-row selected"
                                  : "project-workspace-chapter-row"
                              }
                              onClick={() => {
                                selectChapter(chapter.id);
                              }}
                            >
                              <span className="project-workspace-chapter-number">
                                {chapter.chapter_order}
                              </span>

                              <span className="project-workspace-tree-text">
                                <strong>
                                  {getChapterDisplayName(chapter)}
                                </strong>

                                <small>
                                  {chapter.character_count.toLocaleString()}
                                  字符
                                  {" · "}
                                  {chapter.is_detected
                                    ? "Chapter detected"
                                    : "自动兜底"}
                                </small>
                              </span>
                            </button>
                          ))
                        )}
                      </div>
                    )}
                  </section>
                );
              })}
            </div>
          )}
        </aside>

        <article className="project-workspace-reader">
          <div className="project-workspace-reader-heading">
            <div>
              <span>章节原文</span>
              <h2>{readerTitle}</h2>
            </div>

            {selectedChapter && (
              <span className="project-workspace-position">
                字符
                {" "}
                {selectedChapter.start_character}
                -
                {selectedChapter.end_character}
              </span>
            )}
          </div>

          {!selectedChapterId && (
            <div className="project-workspace-reader-empty">
              <strong>尚未选择章节</strong>
              <p>
                请在左侧目录中选择一个章节查看正文。
              </p>
            </div>
          )}

          {selectedChapterId && chapterLoading && (
            <div className="project-workspace-reader-empty">
              <div className="project-workspace-loader" />
              <strong>正在加载章节正文</strong>
            </div>
          )}

          {selectedChapterId && chapterError && (
            <div className="project-workspace-reader-empty error">
              <strong>章节正文加载失败</strong>
              <p>{chapterError}</p>

              <button
                type="button"
                onClick={retryChapterLoad}
              >
                重试
              </button>
            </div>
          )}

          {selectedChapter
            && !chapterLoading
            && !chapterError && (
              <>
                <div className="project-workspace-reader-meta">
                  <div>
                    <span>源文件</span>
                    <strong>
                      {selectedChapter.source_file_name}
                    </strong>
                  </div>

                  <div>
                    <span>章节顺序</span>
                    <strong>
                      第
                      {" "}
                      {selectedChapter.chapter_order}
                      {" "}
                      章
                    </strong>
                  </div>

                  <div>
                    <span>字符数</span>
                    <strong>
                      {selectedChapter.character_count.toLocaleString()}
                    </strong>
                  </div>

                  <div>
                    <span>识别状态</span>
                    <strong>
                      {selectedChapter.is_detected
                        ? "Chapter title detected"
                        : "使用文件范围兜底"}
                    </strong>
                  </div>

                  <div>
                    <span>字符范围</span>
                    <strong>
                      {selectedChapter.start_character}
                      -
                      {selectedChapter.end_character}
                    </strong>
                  </div>
                </div>

                <pre
                  ref={readerTextRef}
                  className="project-workspace-text"
                >
                  {highlightedChapterText.before}
                  {highlightedChapterText.match ? (
                    <mark
                      ref={highlightedEvidenceRef}
                      className="project-workspace-evidence-highlight"
                    >
                      {highlightedChapterText.match}
                    </mark>
                  ) : null}
                  {highlightedChapterText.after}
                </pre>
              </>
            )}
        </article>
      </section>
    </main>
  );
}


export default ProjectWorkspacePage;
