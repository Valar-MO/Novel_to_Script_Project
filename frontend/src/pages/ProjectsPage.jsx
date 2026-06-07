import {
  useEffect,
  useState,
} from "react";
import {
  useNavigate,
} from "react-router-dom";

import {
  ApiError,
  deleteProject,
  getProjects,
} from "../api/projects";

import "./ProjectsPage.css";


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


function getErrorMessage(error) {
  if (error instanceof ApiError) {
    return error.message;
  }

  if (error instanceof Error) {
    return error.message;
  }

  return "项目列表加载失败，请稍后重试。";
}


function ProjectsPage() {
  const navigate = useNavigate();

  const [projects, setProjects] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState("");
  const [reloadKey, setReloadKey] = useState(0);
  const [deletingProjectId, setDeletingProjectId] = useState(null);

  useEffect(() => {
    const controller = new AbortController();

    setIsLoading(true);
    setErrorMessage("");

    getProjects({
      signal: controller.signal,
    })
      .then((projectItems) => {
        setProjects(
          Array.isArray(projectItems)
            ? projectItems
            : [],
        );
      })
      .catch((error) => {
        if (
          error instanceof DOMException
          && error.name === "AbortError"
        ) {
          return;
        }

        setErrorMessage(getErrorMessage(error));
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setIsLoading(false);
        }
      });

    return () => {
      controller.abort();
    };
  }, [reloadKey]);

  async function handleDeleteProject(project) {
    const confirmed = window.confirm(
      `确定删除项目“${project.project_name}”吗？`
      + "\n\n项目原文、人物关系和已生成剧本都会被删除。"
      + "\n此操作无法撤销。",
    );

    if (!confirmed) {
      return;
    }

    setDeletingProjectId(project.project_id);
    setErrorMessage("");

    try {
      await deleteProject(project.project_id);

      setProjects((currentProjects) => (
        currentProjects.filter(
          (item) => (
            item.project_id !== project.project_id
          ),
        )
      ));
    } catch (error) {
      setErrorMessage(
        getErrorMessage(error),
      );
    } finally {
      setDeletingProjectId(null);
    }
  }

  return (
    <main className="projects-page">
      <header className="projects-page-header">
        <div className="projects-page-navigation">
          <button
            type="button"
            onClick={() => navigate("/")}
          >
            ← 返回首页
          </button>

          <button
            type="button"
            className="primary"
            onClick={() => navigate("/project/new")}
          >
            ＋ 创建新项目
          </button>
        </div>

        <div>
          <p className="projects-page-eyebrow">
            Novel2Script
          </p>
          <h1>已有项目</h1>
          <p className="projects-page-description">
            查看已经保存的小说项目，并重新进入项目工作区。
          </p>
        </div>
      </header>

      <section className="projects-page-content">
        {isLoading && (
          <div className="projects-page-state">
            <div className="projects-page-loader" />
            <strong>正在加载项目列表</strong>
            <p>正在从本地数据库读取已有项目……</p>
          </div>
        )}

        {!isLoading && errorMessage && (
          <div className="projects-page-state error">
            <span className="projects-page-state-icon">!</span>
            <strong>项目列表加载失败</strong>
            <p>{errorMessage}</p>
            <button
              type="button"
              onClick={() => {
                setReloadKey((value) => value + 1);
              }}
            >
              重新加载
            </button>
          </div>
        )}

        {!isLoading
          && !errorMessage
          && projects.length === 0 && (
            <div className="projects-page-state">
              <span className="projects-page-state-icon">＋</span>
              <strong>还没有保存的项目</strong>
              <p>创建第一个小说改编项目后，它会显示在这里。</p>
              <button
                type="button"
                onClick={() => navigate("/project/new")}
              >
                创建新项目
              </button>
            </div>
          )}

        {!isLoading
          && !errorMessage
          && projects.length > 0 && (
            <>
              <div className="projects-page-summary">
                <span>项目数量</span>
                <strong>{projects.length}</strong>
              </div>

              <div className="projects-page-grid">
                {projects.map((project) => (
                  <article
                    key={project.project_id}
                    className="projects-page-card"
                  >
                    <div className="projects-page-card-heading">
                      <div>
                        <span>小说改编项目</span>
                        <h2>{project.project_name}</h2>
                      </div>

                      <span
                        className={`projects-page-status status-${project.status}`}
                      >
                        {getStatusLabel(project.status)}
                      </span>
                    </div>

                    <div className="projects-page-statistics">
                      <div>
                        <span>文件</span>
                        <strong>{project.file_count}</strong>
                      </div>

                      <div>
                        <span>章节</span>
                        <strong>{project.chapter_count}</strong>
                      </div>
                    </div>

                    <div className="projects-page-card-footer">
                      <div>
                        <span>创建时间</span>
                        <strong>
                          {formatDate(project.created_at)}
                        </strong>
                      </div>

                      <div className="projects-page-card-actions">
                        <button
                          type="button"
                          onClick={() => {
                            navigate(
                              `/project/${encodeURIComponent(project.project_id)}`,
                            );
                          }}
                        >
                          打开项目 →
                        </button>

                        <button
                          type="button"
                          className="danger"
                          disabled={
                            deletingProjectId === project.project_id
                          }
                          onClick={() => {
                            handleDeleteProject(project);
                          }}
                        >
                          {deletingProjectId === project.project_id
                            ? "正在删除…"
                            : "删除"}
                        </button>
                      </div>
                    </div>
                  </article>
                ))}
              </div>
            </>
          )}
      </section>
    </main>
  );
}


export default ProjectsPage;
