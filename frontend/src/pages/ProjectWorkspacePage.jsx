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
  getProjectChapter,
  getProjectSummary,
} from "../api/projects";

import "./ProjectWorkspacePage.css";


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


function ProjectWorkspacePage() {
  const { projectId } = useParams();
  const navigate = useNavigate();

  const chapterCacheRef = useRef(new Map());

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

  if (projectLoading) {
    return (
      <main className="project-workspace-state-page">
        <div className="project-workspace-loader" />
        <h1>正在加载项目</h1>
        <p>正在从数据库读取项目摘要……</p>
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
            ? "项目不存在"
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
            ＋ 新建项目
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
                        {fileExpanded ? "▾" : "▸"}
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
                                    ? "章节已识别"
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
                –
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
                重新加载
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
                        ? "章节标题已识别"
                        : "使用文件范围兜底"}
                    </strong>
                  </div>

                  <div>
                    <span>字符范围</span>
                    <strong>
                      {selectedChapter.start_character}
                      –
                      {selectedChapter.end_character}
                    </strong>
                  </div>
                </div>

                <pre className="project-workspace-text">
                  {selectedChapter.text}
                </pre>
              </>
            )}
        </article>
      </section>
    </main>
  );
}


export default ProjectWorkspacePage;
