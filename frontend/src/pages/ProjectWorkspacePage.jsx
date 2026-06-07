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
  appendProjectFiles,
  deleteProject,
  getProjectChunk,
  getProjectChapter,
  getProjectSummary,
} from "../api/projects";
import {
  getActiveNarrativeAnalysisRun,
  getNarrativeAnalysisRun,
  startNarrativeAnalysis,
} from "../api/narrativeAnalysis";
import {
  cancelScriptGeneration,
  getProjectScriptGenerationState,
  getScriptGenerationRun,
  getScriptGenerationScenes,
  regenerateScriptScene,
  startScriptGeneration,
  updateScriptScene,
} from "../api/scriptGeneration";
import {
  buildProjectRelationships,
  createProjectRelationship,
  deleteProjectRelationship,
  getProjectRelationships,
  updateProjectRelationship,
} from "../api/projectRelationships";

import {
  buildProjectCharacters,
  deleteProjectCharacter,
  getLatestProjectCharacters,
  updateProjectCharacterPin,
} from "../api/projectCharacters";

import {
  buildScriptMarkdown,
  buildScriptText,
  downloadTextFile,
  sanitizeFileName,
} from "../utils/scriptExport";

import "./ProjectWorkspacePage.css";


const ANALYSIS_ACTIVE_STATUSES = new Set([
  "queued",
  "running",
]);


function wait(milliseconds) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, milliseconds);
  });
}


async function waitForNarrativeAnalysisCompletion(
  runId,
  onProgress,
) {
  while (true) {
    const run = await getNarrativeAnalysisRun(
      runId,
      {
        includeUnits: false,
      },
    );

    onProgress(run);

    if (
      run.status === "completed"
      || run.status === "partial"
    ) {
      return run;
    }

    if (
      run.status === "failed"
      || run.status === "interrupted"
    ) {
      throw new Error(
        run.error_message
        || "AI 叙事分析未能完成。",
      );
    }

    await wait(2000);
  }
}


const SCRIPT_TERMINAL_STATUSES = new Set([
  "completed",
  "partial",
  "failed",
  "interrupted",
  "cancelled",
]);

const SCRIPT_ACTIVE_STATUSES = new Set([
  "queued",
  "running",
]);

const DEFAULT_VISIBLE_RELATIONSHIP_COUNT = 6;


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


function buildEditedCharacters(text, originalCharacters = []) {
  const originalByName = new Map(
    originalCharacters.map((character) => [
      character.name.trim(),
      character,
    ]),
  );

  const names = [
    ...new Set(
      text
        .split(/[、，,\n]/)
        .map((name) => name.trim())
        .filter(Boolean),
    ),
  ];

  return names.map((name) => ({
    character_id: originalByName.get(name)?.character_id ?? null,
    name,
  }));
}


function getScriptActionLabel(scriptState) {
  switch (scriptState?.suggested_action) {
    case "continue_new":
      return "继续生成新增内容";
    case "continue_remaining":
      return "继续生成剩余内容";
    case "continue_pending":
      return "继续生成待处理内容";
    case "all_generated":
      return "重新生成剧本";
    case "running":
      return "生成中……";
    case "cancelling":
      return "正在取消……";
    case "generate_all":
    default:
      return "生成剧本";
  }
}


function getScriptStartScope(scriptState) {
  switch (scriptState?.suggested_action) {
    case "continue_new":
    case "continue_remaining":
    case "continue_pending":
      return "pending";
    default:
      return "all";
  }
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
  const [isDeletingProject, setIsDeletingProject] = useState(false);

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

  const [analysisResult, setAnalysisResult] = useState(null);
  const [activeMention, setActiveMention] = useState(null);
  const [isGeneratingScript, setIsGeneratingScript] = useState(false);
  const [isCancellingScript, setIsCancellingScript] = useState(false);
  const [scriptState, setScriptState] = useState(null);
  const [scriptRun, setScriptRun] = useState(null);
  const [scriptScenes, setScriptScenes] = useState([]);
  const [scriptError, setScriptError] = useState("");
  const [appendFilesBusy, setAppendFilesBusy] = useState(false);
  const [appendFilesMessage, setAppendFilesMessage] = useState("");
  const [selectedScriptSceneId, setSelectedScriptSceneId] = useState(null);
  const [editingSceneId, setEditingSceneId] = useState(null);
  const [sceneDraft, setSceneDraft] = useState({
    heading: "",
    interiorExterior: "",
    location: "",
    timeOfDay: "",
    charactersText: "",
    scriptText: "",
  });
  const [regeneratingSceneId, setRegeneratingSceneId] = useState(null);
  const [regenerateInstruction, setRegenerateInstruction] = useState("");
  const [scriptModel, setScriptModel] = useState("deepseek-v4-pro");
  const [scriptThinkingEnabled, setScriptThinkingEnabled] = useState(false);
  const [showScriptExportMenu, setShowScriptExportMenu] = useState(false);
  const [relationshipData, setRelationshipData] = useState(null);
  const [characterRun, setCharacterRun] = useState(null);
  const [relationshipError, setRelationshipError] = useState("");
  const [isBuildingRelationships, setIsBuildingRelationships] = useState(false);
  const [showAllRelationships, setShowAllRelationships] = useState(false);
  const [showCharacterManager, setShowCharacterManager] = useState(false);
  const [relationshipBuildStage, setRelationshipBuildStage] = useState("");
  const [editingRelationshipId, setEditingRelationshipId] = useState(null);
  const [relationshipDraft, setRelationshipDraft] = useState({
    sourceCharacterId: "",
    sourceCharacterName: "",
    targetCharacterId: "",
    targetCharacterName: "",
    relationLabel: "",
    relationDescription: "",
    evidenceText: "",
  });

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

  const importantRelationships = (
    relationshipData?.core_relationships || []
  );

  const visibleRelationships = (
    showAllRelationships
      ? importantRelationships
      : importantRelationships.slice(
          0,
          DEFAULT_VISIBLE_RELATIONSHIP_COUNT,
        )
  );

  const hiddenRelationshipCount = Math.max(
    0,
    importantRelationships.length
      - DEFAULT_VISIBLE_RELATIONSHIP_COUNT,
  );

  const importantRelationshipCount = (
    relationshipData?.core_relationships?.length || 0
  );

  async function refreshProjectSummary() {
    const projectData = await getProjectSummary(projectId);
    setProject(projectData);
    setExpandedFileIds(
      new Set(projectData.files.map((file) => file.id)),
    );
    return projectData;
  }

  async function refreshScriptGenerationState() {
    const state = await getProjectScriptGenerationState(projectId);
    setScriptState(state);

    if (state.latest_run) {
      setScriptRun(state.latest_run);

      if (state.latest_run.id) {
        const scenesData = await getScriptGenerationScenes(
          state.latest_run.id,
        );
        setScriptScenes(scenesData.scenes || []);
        setSelectedScriptSceneId((currentId) => (
          currentId || scenesData.scenes?.[0]?.id || null
        ));
      }
    } else {
      setScriptRun(null);
      setScriptScenes([]);
      setSelectedScriptSceneId(null);
    }

    return state;
  }

  async function refreshRelationships({ signal } = {}) {
    const data = await getProjectRelationships(projectId, { signal });
    setRelationshipData(data);
    return data;
  }

  function resetRelationshipDraft() {
    setEditingRelationshipId(null);
    setRelationshipDraft({
      sourceCharacterId: "",
      sourceCharacterName: "",
      targetCharacterId: "",
      targetCharacterName: "",
      relationLabel: "",
      relationDescription: "",
      evidenceText: "",
    });
  }

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

    Promise.all([
      getProjectRelationships(
        projectId,
        {
          signal: controller.signal,
        },
      ),
      getLatestProjectCharacters(
        projectId,
        {
          signal: controller.signal,
        },
      ),
    ])
      .then(([
        relationships,
        latestCharacterRun,
      ]) => {
        setRelationshipData(relationships);
        setCharacterRun(latestCharacterRun);
      })
      .catch((error) => {
        if (
          error instanceof DOMException
          && error.name === "AbortError"
        ) {
          return;
        }

        setRelationshipError(
          getErrorMessage(
            error,
            "读取人物关系失败。",
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

    getProjectScriptGenerationState(projectId, {
      signal: controller.signal,
    })
      .then(async (state) => {
        setScriptState(state);
        setIsGeneratingScript(
          SCRIPT_ACTIVE_STATUSES.has(state.latest_run?.status),
        );
        setIsCancellingScript(
          Boolean(
            state.latest_run?.cancel_requested_at
            && !state.latest_run?.cancelled_at,
          ),
        );

        if (!state.latest_run) {
          setScriptRun(null);
          setScriptScenes([]);
          setSelectedScriptSceneId(null);
          return;
        }

        setScriptRun(state.latest_run);

        const scenesData = await getScriptGenerationScenes(
          state.latest_run.id,
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
            "读取剧本生成状态失败。",
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
          setIsCancellingScript(false);
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

  async function handleBuildRelationships() {
    if (isBuildingRelationships) {
      return;
    }

    setIsBuildingRelationships(true);
    setRelationshipError("");
    setRelationshipBuildStage("正在检查分析任务");
    setAnalysisResult(null);

    try {
      let activeRun = await getActiveNarrativeAnalysisRun(
        projectId,
      );

      let runId;

      if (
        activeRun
        && ANALYSIS_ACTIVE_STATUSES.has(activeRun.status)
      ) {
        runId = activeRun.id || activeRun.run_id;
        setAnalysisResult(activeRun);
      } else {
        setRelationshipBuildStage("正在启动文本分析");

        const startedRun = await startNarrativeAnalysis(
          projectId,
          {
            maxChunks: null,
            forceReanalyze: false,
          },
        );

        runId = startedRun.run_id;

        setAnalysisResult({
          id: runId,
          ...startedRun,
        });
      }

      setRelationshipBuildStage("正在分析小说文本");

      const completedAnalysis = (
        await waitForNarrativeAnalysisCompletion(
          runId,
          (runDetail) => {
            setAnalysisResult(runDetail);
          },
        )
      );

      setRelationshipBuildStage("正在整理项目人物");

      const builtCharacterRun = await buildProjectCharacters(
        projectId,
        completedAnalysis.id,
      );

      if (!builtCharacterRun?.characters?.length) {
        throw new Error(
          "没有识别到可用人物，暂时无法生成人物关系。",
        );
      }

      setCharacterRun(builtCharacterRun);

      setRelationshipBuildStage("正在筛选重要关系");

      const relationships = await buildProjectRelationships(
        projectId,
        builtCharacterRun.id,
      );

      setRelationshipData(relationships);
      setShowAllRelationships(false);
      resetRelationshipDraft();
    } catch (error) {
      setRelationshipError(
        getErrorMessage(
          error,
          "分析人物关系失败。",
        ),
      );
    } finally {
      setIsBuildingRelationships(false);
      setRelationshipBuildStage("");
    }
  }

  async function handleToggleCharacterPin(character) {
    setRelationshipError("");
    setIsBuildingRelationships(true);
    setRelationshipBuildStage("正在更新核心人物");

    try {
      const updatedCharacterRun = (
        await updateProjectCharacterPin(
          character.id,
          !character.is_user_pinned,
        )
      );

      setCharacterRun(updatedCharacterRun);

      setRelationshipBuildStage("正在重新筛选重要关系");

      const relationships = await buildProjectRelationships(
        projectId,
        updatedCharacterRun.id,
      );

      setRelationshipData(relationships);
    } catch (error) {
      setRelationshipError(
        getErrorMessage(
          error,
          "更新核心人物失败。",
        ),
      );
    } finally {
      setIsBuildingRelationships(false);
      setRelationshipBuildStage("");
    }
  }

  async function handleDeleteOrdinaryCharacter(
    character,
  ) {
    const confirmed = window.confirm(
      `确定删除普通人物“${character.canonical_name}”吗？`
      + "\n删除后，该人物以及与其相关的关系将不再展示。",
    );

    if (!confirmed) {
      return;
    }

    setRelationshipError("");
    setIsBuildingRelationships(true);
    setRelationshipBuildStage("正在删除普通人物");

    try {
      const updatedCharacterRun = (
        await deleteProjectCharacter(character.id)
      );

      setCharacterRun(updatedCharacterRun);

      const relationships = await getProjectRelationships(
        projectId,
      );

      setRelationshipData(relationships);
    } catch (error) {
      setRelationshipError(
        getErrorMessage(
          error,
          "删除普通人物失败。",
        ),
      );
    } finally {
      setIsBuildingRelationships(false);
      setRelationshipBuildStage("");
    }
  }

  async function handleToggleCharacterManager() {
    if (showCharacterManager) {
      setShowCharacterManager(false);
      return;
    }

    setRelationshipError("");

    try {
      if (!characterRun) {
        const latestCharacterRun = (
          await getLatestProjectCharacters(projectId)
        );

        setCharacterRun(latestCharacterRun);
      }

      setShowCharacterManager(true);
    } catch (error) {
      setRelationshipError(
        getErrorMessage(
          error,
          "读取项目人物失败。",
        ),
      );
    }
  }

  function handleStartRelationshipEdit(relationship) {
    setEditingRelationshipId(relationship.id);
    setRelationshipDraft({
      sourceCharacterId: relationship.source_character_id,
      sourceCharacterName: relationship.source_character_name,
      targetCharacterId: relationship.target_character_id,
      targetCharacterName: relationship.target_character_name,
      relationLabel: relationship.relation_label,
      relationDescription: relationship.relation_description || "",
      evidenceText: relationship.evidence_text || "",
    });
  }

  function handleStartRelationshipCreate() {
    const [firstCharacter, secondCharacter] = (
      relationshipData?.core_characters || []
    );
    setEditingRelationshipId("new");
    setRelationshipDraft({
      sourceCharacterId: firstCharacter?.character_id || "",
      sourceCharacterName: firstCharacter?.canonical_name || "",
      targetCharacterId: secondCharacter?.character_id || "",
      targetCharacterName: secondCharacter?.canonical_name || "",
      relationLabel: "",
      relationDescription: "",
      evidenceText: "",
    });
  }

  function updateRelationshipDraftField(fieldName, value) {
    setRelationshipDraft((currentDraft) => ({
      ...currentDraft,
      [fieldName]: value,
    }));
  }

  function updateRelationshipDraftCharacter(fieldName, characterId) {
    const character = (relationshipData?.core_characters || [])
      .find((item) => item.character_id === characterId);

    setRelationshipDraft((currentDraft) => ({
      ...currentDraft,
      [fieldName]: characterId,
      [fieldName === "sourceCharacterId"
        ? "sourceCharacterName"
        : "targetCharacterName"]: character?.canonical_name || "",
    }));
  }

  async function handleSaveRelationship() {
    setRelationshipError("");

    try {
      if (editingRelationshipId === "new") {
        await createProjectRelationship(
          projectId,
          {
            source_character_id: relationshipDraft.sourceCharacterId,
            source_character_name: relationshipDraft.sourceCharacterName,
            target_character_id: relationshipDraft.targetCharacterId,
            target_character_name: relationshipDraft.targetCharacterName,
            relation_label: relationshipDraft.relationLabel,
            relation_description: relationshipDraft.relationDescription,
            evidence_text: relationshipDraft.evidenceText,
          },
        );
      } else {
        await updateProjectRelationship(
          editingRelationshipId,
          {
            relation_label: relationshipDraft.relationLabel,
            relation_description: relationshipDraft.relationDescription,
            evidence_text: relationshipDraft.evidenceText,
          },
        );
      }

      await refreshRelationships();
      resetRelationshipDraft();
    } catch (error) {
      setRelationshipError(
        getErrorMessage(
          error,
          "保存人物关系失败。",
        ),
      );
    }
  }

  async function handleDeleteRelationship(relationship) {
    const confirmed = window.confirm(
      `确定删除关系「${relationship.relation_label}」吗？`,
    );

    if (!confirmed) {
      return;
    }

    setRelationshipError("");

    try {
      await deleteProjectRelationship(relationship.id);
      await refreshRelationships();
      if (editingRelationshipId === relationship.id) {
        resetRelationshipDraft();
      }
    } catch (error) {
      setRelationshipError(
        getErrorMessage(
          error,
          "删除人物关系失败。",
        ),
      );
    }
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
          scope: getScriptStartScope(scriptState),
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
      await refreshScriptGenerationState();
    } catch (error) {
      try {
        const state = await refreshScriptGenerationState();
        const latestRun = state?.latest_run;

        if (
          latestRun
          && SCRIPT_ACTIVE_STATUSES.has(latestRun.status)
        ) {
          setIsGeneratingScript(true);
          setIsCancellingScript(
            Boolean(
              latestRun.cancel_requested_at
              && !latestRun.cancelled_at,
            ),
          );
          setScriptError("");
          return;
        }
      } catch {
        // Fall through to the original start error.
      }

      setIsGeneratingScript(false);
      setScriptError(
        getErrorMessage(
          error,
          "启动剧本生成失败。",
        ),
      );
    }
  }

  async function handleCancelScriptGeneration() {
    const runId = scriptRun?.id || scriptRun?.run_id;

    if (!runId) {
      return;
    }

    const confirmed = window.confirm(
      "确定要取消当前剧本生成吗？当前正在处理的文本块会先完成，然后停止后续生成。",
    );

    if (!confirmed) {
      return;
    }

    setIsCancellingScript(true);
    setScriptError("");

    try {
      await cancelScriptGeneration(runId);
      await refreshScriptGenerationState();
    } catch (error) {
      setScriptError(
        getErrorMessage(
          error,
          "取消剧本生成失败。",
        ),
      );
      setIsCancellingScript(false);
    }
  }

  function handleSelectScriptScene(scene) {
    setSelectedScriptSceneId(scene.id);
    setActiveMention(null);
  }

  async function handleViewScriptSceneSource(scene) {
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

  function handleStartSceneEdit(scene) {
    setEditingSceneId(scene.id);
    setSceneDraft({
      heading: scene.heading,
      interiorExterior: scene.interior_exterior,
      location: scene.location,
      timeOfDay: scene.time_of_day,
      charactersText: (scene.characters || [])
        .map((character) => character.name)
        .join("、"),
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
      charactersText: "",
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
          characters: buildEditedCharacters(
            sceneDraft.charactersText,
            scene.characters,
          ),
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
    if (scene.is_user_edited) {
      const confirmed = window.confirm(
        "当前场景包含人工修改，重新生成将覆盖这些内容。",
      );

      if (!confirmed) {
        return;
      }
    }

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

  async function handleDeleteCurrentProject() {
    const confirmed = window.confirm(
      `确定删除项目“${project.project_name}”吗？`
      + "\n\n小说原文、人物关系和已生成剧本都会被删除。"
      + "\n此操作无法撤销。",
    );

    if (!confirmed) {
      return;
    }

    setIsDeletingProject(true);
    setProjectError("");

    try {
      await deleteProject(projectId);

      navigate("/projects", {
        replace: true,
      });
    } catch (error) {
      setProjectError(
        getErrorMessage(
          error,
          "删除项目失败。",
        ),
      );

      setIsDeletingProject(false);
    }
  }

  function handleExportScript(format) {
    if (scriptScenes.length === 0) {
      setScriptError(
        "当前没有可导出的剧本场景。",
      );
      return;
    }

    setScriptError("");
    setShowScriptExportMenu(false);

    const baseFileName = sanitizeFileName(
      project?.project_name || "剧本",
    );

    if (format === "markdown") {
      const content = buildScriptMarkdown(
        project?.project_name,
        scriptScenes,
      );

      downloadTextFile(
        `${baseFileName}.md`,
        content,
        "text/markdown",
      );

      return;
    }

    const content = buildScriptText(
      project?.project_name,
      scriptScenes,
    );

    downloadTextFile(
      `${baseFileName}.txt`,
      content,
      "text/plain",
    );
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

  async function handleMentionClick(mention) {
    setActiveMention(mention);

    const sourceChunkId = mention.source_chunk_id || mention.chunk_id;
    if (!sourceChunkId) {
      return;
    }

    try {
      const chunk = await getProjectChunk(projectId, sourceChunkId);

      if (chunk?.chapter_id && chunk.chapter_id !== selectedChapterId) {
        setSelectedChapterId(chunk.chapter_id);
        setChapterReloadKey(0);
      }
    } catch {
      // Highlight still works if the current chapter already contains evidence.
    }
  }

  const appendFilesControl = (
    <div className="project-workspace-append-files">
      <label
        className={
          appendFilesBusy || isGeneratingScript || isCancellingScript
            ? "project-workspace-append-files-button disabled"
            : "project-workspace-append-files-button"
        }
      >
        <span>
          {appendFilesBusy ? "正在追加..." : "追加小说文件"}
        </span>
        <small>用于生成新增内容</small>
        <input
          type="file"
          accept=".txt"
          multiple
          disabled={appendFilesBusy || isGeneratingScript || isCancellingScript}
          onChange={async (event) => {
            const selectedFiles = Array.from(event.target.files || []);
            event.target.value = "";

            if (selectedFiles.length === 0) {
              return;
            }

            setAppendFilesBusy(true);
            setAppendFilesMessage("");
            setScriptError("");

            try {
              const result = await appendProjectFiles(
                projectId,
                selectedFiles,
              );
              await refreshProjectSummary();
              await refreshScriptGenerationState();
              setAppendFilesMessage(
                `已追加 ${result.added_file_count} 个文件，新增 ${result.added_chunk_count} 个文本块。`,
              );
            } catch (error) {
              setAppendFilesMessage(
                getErrorMessage(
                  error,
                  "追加文件失败。",
                ),
              );
            } finally {
              setAppendFilesBusy(false);
            }
          }}
        />
      </label>

      {appendFilesMessage && (
        <p className="project-workspace-append-files-message">
          {appendFilesMessage}
        </p>
      )}
    </div>
  );

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

          <div className="project-workspace-header-actions">
            <span className="project-workspace-status">
              {getStatusLabel(project.status)}
            </span>

            <button
              type="button"
              className="project-workspace-danger-button"
              disabled={isDeletingProject}
              onClick={handleDeleteCurrentProject}
            >
              {isDeletingProject ? "正在删除…" : "删除项目"}
            </button>
          </div>
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
      </header>

      <section className="project-workspace-tools">
        <section className="project-workspace-relationships">
          <div className="project-workspace-analysis-heading">
            <div className="project-workspace-analysis-heading-card">
              <strong>人物关系</strong>
              <span>生成可编辑的核心人物关系，供后续剧本生成参考</span>
            </div>

            <div className="project-workspace-analysis-actions">
              <button
                type="button"
                className="project-workspace-analysis-button"
                disabled={isBuildingRelationships}
                onClick={handleBuildRelationships}
              >
                {isBuildingRelationships
                  ? "正在分析…"
                  : relationshipData?.relationships?.length
                    ? "重新分析人物关系"
                    : "分析人物关系"}
              </button>

              <button
                type="button"
                className="project-workspace-analysis-button secondary"
                disabled={
                  isBuildingRelationships
                  || !relationshipData?.core_characters?.length
                }
                onClick={handleStartRelationshipCreate}
              >
                新增关系
              </button>

              <button
                type="button"
                className="project-workspace-analysis-button secondary"
                disabled={
                  isBuildingRelationships
                  || !characterRun?.characters?.length
                }
                onClick={handleToggleCharacterManager}
              >
                {showCharacterManager
                  ? "收起人物管理"
                  : "管理核心人物"}
              </button>
            </div>
          </div>

          {relationshipError && (
            <p className="project-workspace-analysis-error">
              {relationshipError}
            </p>
          )}

          {isBuildingRelationships && (
            <div className="project-workspace-relationship-progress">
              <strong>
                {relationshipBuildStage || "正在处理人物关系"}
              </strong>

              {analysisResult?.total_chunks > 0 && (
                <span>
                  {analysisResult.processed_chunks ?? 0}
                  {" / "}
                  {analysisResult.total_chunks}
                  {" 个文本块"}
                </span>
              )}
            </div>
          )}

          <div className="project-workspace-relationship-summary">
            <div>
              <span>核心人物</span>
              <strong>
                {relationshipData?.core_characters?.length || 0}
              </strong>
            </div>

            <div>
              <span>重要关系</span>
              <strong>
                {relationshipData?.core_relationships?.length || 0}
              </strong>
            </div>
          </div>

          {relationshipData?.core_characters?.length > 0 && (
            <div className="project-workspace-core-characters">
              {relationshipData.core_characters.map((character) => (
                <div
                  key={character.character_id}
                  className={
                    character.is_user_pinned
                      ? "project-workspace-core-character pinned"
                      : "project-workspace-core-character"
                  }
                  title={
                    character.is_user_pinned
                      ? "用户固定的核心人物"
                      : "系统筛选的核心人物"
                  }
                >
                  <strong>{character.canonical_name}</strong>
                </div>
              ))}
            </div>
          )}

          {showCharacterManager && (
            <section className="project-workspace-character-manager">
              <div className="project-workspace-character-manager-heading">
                <div>
                  <strong>管理核心人物</strong>
                  <span>
                    固定的人物会始终作为核心人物参与关系筛选
                  </span>
                </div>
              </div>

              {(characterRun?.characters || []).length > 0 ? (
                <div className="project-workspace-character-manager-list">
                  {characterRun.characters.map((character) => {
                    const isCore = (
                      relationshipData?.core_characters || []
                    ).some(
                      (coreCharacter) => (
                        coreCharacter.character_id
                        === character.character_id
                      ),
                    );

                    const inputQuality = character.input_quality || {};

                    const chunkCount = Number(
                      inputQuality.chunk_count || 0,
                    );

                    const mentionCount = Number(
                      inputQuality.mention_count
                      || character.evidence_count
                      || 0,
                    );

                    let statusLabel = "普通人物";

                    if (character.is_user_pinned) {
                      statusLabel = "已固定为核心";
                    } else if (isCore) {
                      statusLabel = "自动核心";
                    }

                    const canDelete = (
                      !character.is_user_pinned
                      && !isCore
                    );

                    return (
                      <div
                        key={character.id}
                        className="project-workspace-character-manager-item"
                      >
                        <input
                          type="checkbox"
                          checked={Boolean(character.is_user_pinned)}
                          disabled={isBuildingRelationships}
                          onChange={() => {
                            handleToggleCharacterPin(character);
                          }}
                        />

                        <span className="project-workspace-character-manager-name">
                          <strong>{character.canonical_name}</strong>

                          <small>
                            出现于 {chunkCount} 个文本块，共
                            {" "}
                            {mentionCount}
                            {" "}
                            次提及
                          </small>
                        </span>

                        <span className="project-workspace-character-manager-status">
                          {statusLabel}
                        </span>

                        {canDelete && (
                          <button
                            type="button"
                            className="project-workspace-character-delete-button"
                            disabled={isBuildingRelationships}
                            onClick={() => {
                              handleDeleteOrdinaryCharacter(character);
                            }}
                          >
                            删除
                          </button>
                        )}
                      </div>
                    );
                  })}
                </div>
              ) : (
                <p className="project-workspace-mention-empty">
                  尚未生成人物表。
                </p>
              )}
            </section>
          )}

          {importantRelationships.length > 0 ? (
            <>
              <div className="project-workspace-relationship-list-heading">
                <strong>重要关系</strong>

                <span>
                  当前显示 {visibleRelationships.length}
                  {" / "}
                  {importantRelationships.length}
                </span>
              </div>

              <div className="project-workspace-relationship-list">
                {visibleRelationships.map((relationship) => (
                  <article
                    key={relationship.id}
                    className="project-workspace-relationship-card"
                  >
                    <div>
                      <strong>
                        {relationship.source_character_name}
                        {" — "}
                        {relationship.relation_label}
                        {" — "}
                        {relationship.target_character_name}
                      </strong>
                      <span>
                        {relationship.source_type === "user"
                          ? "人工确认"
                          : `AI 提取 · ${relationship.evidence_count} 条证据`}
                      </span>
                    </div>

                    {relationship.relation_description && (
                      <p>{relationship.relation_description}</p>
                    )}

                    <div className="project-workspace-relationship-actions">
                      {relationship.evidence_text && (
                        <button
                          type="button"
                          onClick={() => handleMentionClick(relationship)}
                        >
                          查看原文
                        </button>
                      )}

                      <button
                        type="button"
                        onClick={() => handleStartRelationshipEdit(relationship)}
                      >
                        编辑
                      </button>

                      <button
                        type="button"
                        onClick={() => handleDeleteRelationship(relationship)}
                      >
                        删除
                      </button>
                    </div>
                  </article>
                ))}
              </div>

              {importantRelationships.length
                > DEFAULT_VISIBLE_RELATIONSHIP_COUNT && (
                <div className="project-workspace-relationship-expand">
                  <button
                    type="button"
                    className="project-workspace-relationship-expand-button"
                    onClick={() => {
                      setShowAllRelationships(
                        (currentValue) => !currentValue,
                      );
                    }}
                  >
                    {showAllRelationships
                      ? "收起重要关系"
                      : `展开其余 ${hiddenRelationshipCount} 条关系`}
                  </button>
                </div>
              )}
            </>
          ) : (
            <p className="project-workspace-mention-empty">
              暂无重要人物关系。
            </p>
          )}

          <div
            className={
              importantRelationshipCount > 0
                ? "project-workspace-relationship-impact"
                : "project-workspace-relationship-impact empty"
            }
          >
            <div className="project-workspace-relationship-impact-icon">
              ↓
            </div>

            <div className="project-workspace-relationship-impact-content">
              {importantRelationshipCount > 0 ? (
                <>
                  <strong>
                    已有 {importantRelationshipCount} 条人物关系可用于剧本生成
                  </strong>

                  <p>
                    系统会参考人物身份、称呼、对白语气和互动方式，
                    以保持不同场景中的人物关系一致。
                  </p>

                  <span>
                    修改后需要重新生成场景，新的关系才会反映在剧本中。
                  </span>
                </>
              ) : (
                <>
                  <strong>尚未建立可用的人物关系</strong>

                  <p>
                    剧本仍可以根据小说原文生成，但人物称呼和互动方式
                    将主要由模型根据当前文本判断。
                  </p>
                </>
              )}
            </div>
          </div>

          {editingRelationshipId && (
            <div className="project-workspace-relationship-editor">
              <h4>
                {editingRelationshipId === "new"
                  ? "新增关系"
                  : "编辑关系"}
              </h4>

              {editingRelationshipId === "new" && (
                <div className="project-workspace-relationship-editor-grid">
                  <label>
                    人物一
                    <select
                      value={relationshipDraft.sourceCharacterId}
                      onChange={(event) => {
                        updateRelationshipDraftCharacter(
                          "sourceCharacterId",
                          event.target.value,
                        );
                      }}
                    >
                      {(relationshipData?.core_characters || []).map(
                        (character) => (
                          <option
                            key={character.character_id}
                            value={character.character_id}
                          >
                            {character.canonical_name}
                          </option>
                        ),
                      )}
                    </select>
                  </label>

                  <label>
                    人物二
                    <select
                      value={relationshipDraft.targetCharacterId}
                      onChange={(event) => {
                        updateRelationshipDraftCharacter(
                          "targetCharacterId",
                          event.target.value,
                        );
                      }}
                    >
                      {(relationshipData?.core_characters || []).map(
                        (character) => (
                          <option
                            key={character.character_id}
                            value={character.character_id}
                          >
                            {character.canonical_name}
                          </option>
                        ),
                      )}
                    </select>
                  </label>
                </div>
              )}

              <label>
                关系
                <input
                  value={relationshipDraft.relationLabel}
                  onChange={(event) => {
                    updateRelationshipDraftField(
                      "relationLabel",
                      event.target.value,
                    );
                  }}
                  placeholder="例如：兄弟、名义师徒，彼此提防"
                />
              </label>

              <label>
                说明
                <textarea
                  value={relationshipDraft.relationDescription}
                  onChange={(event) => {
                    updateRelationshipDraftField(
                      "relationDescription",
                      event.target.value,
                    );
                  }}
                  rows={3}
                />
              </label>

              <label>
                原文证据
                <textarea
                  value={relationshipDraft.evidenceText}
                  onChange={(event) => {
                    updateRelationshipDraftField(
                      "evidenceText",
                      event.target.value,
                    );
                  }}
                  rows={3}
                />
              </label>

              <div className="project-workspace-script-actions">
                <button
                  type="button"
                  className="project-workspace-analysis-button"
                  onClick={handleSaveRelationship}
                >
                  保存关系
                </button>

                <button
                  type="button"
                  onClick={resetRelationshipDraft}
                >
                  取消
                </button>
              </div>
            </div>
          )}
        </section>

        <section className="project-workspace-script">
          <div className="project-workspace-analysis-heading">
            <div className="project-workspace-analysis-heading-card">
              <div className="project-workspace-section-title-row">
                <strong>剧本场景生成</strong>

                <span
                  className={
                    importantRelationshipCount > 0
                      ? "project-workspace-relationship-status active"
                      : "project-workspace-relationship-status"
                  }
                >
                  {importantRelationshipCount > 0
                    ? `已关联 ${importantRelationshipCount} 条人物关系`
                    : "未关联人物关系"}
                </span>
              </div>

              <span>
                根据小说原文和项目人物关系生成可编辑剧本场景
              </span>
            </div>

            <div className="project-workspace-analysis-actions">
              <button
                type="button"
                className="project-workspace-analysis-button"
                disabled={
                  isGeneratingScript
                  || isCancellingScript
                }
                onClick={handleStartScriptGeneration}
              >
                {getScriptActionLabel(scriptState)}
              </button>

              {SCRIPT_ACTIVE_STATUSES.has(scriptRun?.status) && (
                <button
                  type="button"
                  className="project-workspace-analysis-button secondary"
                  disabled={isCancellingScript}
                  onClick={handleCancelScriptGeneration}
                >
                  {isCancellingScript ? "正在取消……" : "取消生成"}
                </button>
              )}

              <div className="project-workspace-export-menu">
                <button
                  type="button"
                  className="project-workspace-analysis-button secondary"
                  disabled={scriptScenes.length === 0}
                  onClick={() => {
                    if (scriptScenes.length === 0) {
                      setScriptError("当前没有可导出的剧本场景。");
                      return;
                    }

                    setScriptError("");
                    setShowScriptExportMenu((currentValue) => !currentValue);
                  }}
                >
                  导出
                </button>

                {showScriptExportMenu && scriptScenes.length > 0 && (
                  <div className="project-workspace-export-menu-panel">
                    <button
                      type="button"
                      onClick={() => {
                        handleExportScript("text");
                      }}
                    >
                      导出为 TXT
                    </button>

                    <button
                      type="button"
                      onClick={() => {
                        handleExportScript("markdown");
                      }}
                    >
                      导出为 Markdown
                    </button>
                  </div>
                )}
              </div>
            </div>
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
                disabled={isGeneratingScript || isCancellingScript}
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
                disabled={isGeneratingScript || isCancellingScript}
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
                          <span>内/外景</span>
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
                        <span>出场人物</span>
                        <input
                          value={sceneDraft.charactersText}
                          placeholder="例如：韩立、舞岩、韩母"
                          onChange={(event) => {
                            setSceneDraft((currentDraft) => ({
                              ...currentDraft,
                              charactersText: event.target.value,
                            }));
                          }}
                        />
                      </label>

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
                            onClick={() => handleViewScriptSceneSource(selectedScriptScene)}
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
                    </>
                  )}
                </article>
              )}
            </div>
          )}
        </section>
      </section>

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

          <div className="project-workspace-sidebar-footer">
            {appendFilesControl}
          </div>
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
