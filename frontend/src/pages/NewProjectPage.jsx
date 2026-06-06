import {
    useMemo,
    useState,
  } from "react";
  
  import {
    useNavigate,
  } from "react-router-dom";
  
  import {
    ApiError,
    uploadProject,
  } from "../api/projects";
  
  import "./NewProjectPage.css";
  
  
  const MAX_FILE_COUNT = 50;
  const MAX_SINGLE_FILE_SIZE = 10 * 1024 * 1024;
  const MAX_TOTAL_FILE_SIZE = 20 * 1024 * 1024;
  
  
  function createFileId(file) {
    if (
      typeof crypto !== "undefined"
      && typeof crypto.randomUUID === "function"
    ) {
      return crypto.randomUUID();
    }
  
    return [
      file.name,
      file.size,
      file.lastModified,
      Math.random().toString(16).slice(2),
    ].join("-");
  }
  
  
  function getDuplicateKey(file) {
    return [
      file.name.toLowerCase(),
      file.size,
      file.lastModified,
    ].join("::");
  }
  
  
  function naturalFileCompare(firstItem, secondItem) {
    return firstItem.file.name.localeCompare(
      secondItem.file.name,
      "zh-CN",
      {
        numeric: true,
        sensitivity: "base",
      },
    );
  }
  
  
  function countLines(text) {
    if (!text) {
      return 0;
    }
  
    return text.split(/\r\n|\r|\n/).length;
  }
  
  
  async function readUtf8Text(file) {
    const buffer = await file.arrayBuffer();
  
    const decoder = new TextDecoder("utf-8", {
      fatal: true,
    });
  
    return decoder.decode(buffer).replace(/^\uFEFF/, "");
  }
  
  
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
  
  
  function NewProjectPage() {
    const navigate = useNavigate();
  
    const [projectName, setProjectName] = useState("");
    const [fileItems, setFileItems] = useState([]);
    const [selectedFileId, setSelectedFileId] = useState(null);
  
    const [validationMessages, setValidationMessages] = useState([]);
    const [uploadError, setUploadError] = useState("");
    const [isReadingFiles, setIsReadingFiles] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
  
    const selectedFileItem = useMemo(
      () => (
        fileItems.find(
          (item) => item.id === selectedFileId,
        ) || null
      ),
      [fileItems, selectedFileId],
    );
  
    const projectSummary = useMemo(
      () => fileItems.reduce(
        (summary, item) => ({
          sizeBytes: summary.sizeBytes + item.file.size,
          characterCount: (
            summary.characterCount + item.characterCount
          ),
          lineCount: summary.lineCount + item.lineCount,
        }),
        {
          sizeBytes: 0,
          characterCount: 0,
          lineCount: 0,
        },
      ),
      [fileItems],
    );
  
    async function addFiles(incomingFiles) {
      if (incomingFiles.length === 0) {
        return;
      }
  
      setIsReadingFiles(true);
      setUploadError("");
  
      try {
        const messages = [];
  
        /*
         * existingItems 保持用户当前已经调整好的顺序。
         * newItems 只保存本次新加入并通过校验的文件。
         */
        const existingItems = [...fileItems];
        const newItems = [];
  
        const duplicateKeys = new Set(
          existingItems.map(
            (item) => getDuplicateKey(item.file),
          ),
        );
  
        let runningSize = existingItems.reduce(
          (total, item) => total + item.file.size,
          0,
        );
  
        for (const file of incomingFiles) {
          const currentFileCount = (
            existingItems.length + newItems.length
          );
  
          if (currentFileCount >= MAX_FILE_COUNT) {
            messages.push(
              `最多只能添加 ${MAX_FILE_COUNT} 个文件。`,
            );
            break;
          }
  
          if (!file.name.toLowerCase().endsWith(".txt")) {
            messages.push(
              `“${file.name}”不是 TXT 文件，已忽略。`,
            );
            continue;
          }
  
          if (file.size === 0) {
            messages.push(
              `“${file.name}”为空文件，已忽略。`,
            );
            continue;
          }
  
          if (file.size > MAX_SINGLE_FILE_SIZE) {
            messages.push(
              `“${file.name}”超过 10 MB，已忽略。`,
            );
            continue;
          }
  
          if (
            runningSize + file.size
            > MAX_TOTAL_FILE_SIZE
          ) {
            messages.push(
              `加入“${file.name}”后总大小将超过 20 MB，已忽略。`,
            );
            continue;
          }
  
          const duplicateKey = getDuplicateKey(file);
  
          if (duplicateKeys.has(duplicateKey)) {
            messages.push(
              `“${file.name}”已存在，未重复添加。`,
            );
            continue;
          }
  
          try {
            const text = await readUtf8Text(file);
  
            if (!text.trim()) {
              messages.push(
                `“${file.name}”不包含有效文本，已忽略。`,
              );
              continue;
            }
  
            newItems.push({
              id: createFileId(file),
              file,
              text,
              characterCount: text.length,
              lineCount: countLines(text),
            });
  
            duplicateKeys.add(duplicateKey);
            runningSize += file.size;
          } catch {
            messages.push(
              `“${file.name}”不是有效的 UTF-8 编码，已忽略。`,
            );
          }
        }
  
        /*
         * 只对本次新增的一批文件进行自然排序。
         * existingItems 的人工顺序完全不变。
         */
        newItems.sort(naturalFileCompare);
  
        const combinedItems = [
          ...existingItems,
          ...newItems,
        ];
  
        setFileItems(combinedItems);
        setValidationMessages(messages);
  
        if (
          !selectedFileId
          && combinedItems.length > 0
        ) {
          setSelectedFileId(combinedItems[0].id);
        }
      } finally {
        setIsReadingFiles(false);
      }
    }
  
    async function handleFileInput(event) {
      const incomingFiles = Array.from(
        event.target.files || [],
      );
  
      event.target.value = "";
  
      await addFiles(incomingFiles);
    }
  
    function moveFile(index, direction) {
      const targetIndex = index + direction;
  
      if (
        targetIndex < 0
        || targetIndex >= fileItems.length
      ) {
        return;
      }
  
      setFileItems((currentItems) => {
        const nextItems = [...currentItems];
  
        [
          nextItems[index],
          nextItems[targetIndex],
        ] = [
          nextItems[targetIndex],
          nextItems[index],
        ];
  
        return nextItems;
      });
    }
  
    function removeFile(fileId) {
      const removedIndex = fileItems.findIndex(
        (item) => item.id === fileId,
      );
  
      const nextItems = fileItems.filter(
        (item) => item.id !== fileId,
      );
  
      setFileItems(nextItems);
  
      if (selectedFileId === fileId) {
        const nextSelectedItem = (
          nextItems[removedIndex]
          || nextItems[removedIndex - 1]
          || nextItems[0]
          || null
        );
  
        setSelectedFileId(
          nextSelectedItem?.id || null,
        );
      }
    }
  
    function clearFiles() {
      setFileItems([]);
      setSelectedFileId(null);
      setValidationMessages([]);
      setUploadError("");
    }
  
    async function handleSubmit(event) {
      event.preventDefault();
  
      const normalizedProjectName = projectName.trim();
  
      if (!normalizedProjectName) {
        setUploadError("请输入项目名称。");
        return;
      }
  
      if (fileItems.length === 0) {
        setUploadError("请至少上传一个 TXT 文件。");
        return;
      }
  
      setIsUploading(true);
      setUploadError("");
  
      try {
        const result = await uploadProject(
          normalizedProjectName,
          fileItems.map((item) => item.file),
        );
  
        if (!result?.project_id) {
          throw new Error(
            "后端没有返回有效的 project_id。",
          );
        }
  
        navigate(
          `/project/${encodeURIComponent(result.project_id)}`,
          {
            replace: true,
          },
        );
      } catch (error) {
        if (error instanceof ApiError) {
          setUploadError(error.message);
        } else {
          setUploadError(
            error instanceof Error
              ? error.message
              : "项目上传失败，请稍后重试。",
          );
        }
      } finally {
        setIsUploading(false);
      }
    }
  
    return (
      <main className="new-project-page">
        <header className="new-project-header">
          <button
            type="button"
            className="new-project-back-button"
            onClick={() => navigate("/")}
          >
            ← 返回首页
          </button>
  
          <div>
            <p className="new-project-eyebrow">
              Novel2Script
            </p>
  
            <h1>创建小说改编项目</h1>
  
            <p>
              上传并排序小说 TXT 文件。后端将自动完成
              文本预处理、章节识别、分块和项目保存。
            </p>
          </div>
        </header>
  
        <form
          className="new-project-form"
          onSubmit={handleSubmit}
        >
          <section className="new-project-card">
            <div className="new-project-section-heading">
              <div>
                <span className="new-project-step">
                  01
                </span>
  
                <h2>项目基本信息</h2>
              </div>
            </div>
  
            <label
              className="new-project-field"
              htmlFor="project-name"
            >
              <span>项目名称</span>
  
              <input
                id="project-name"
                type="text"
                value={projectName}
                maxLength={100}
                placeholder="例如：山边小村影视改编"
                disabled={isUploading}
                onChange={(event) => {
                  setProjectName(event.target.value);
                  setUploadError("");
                }}
              />
            </label>
          </section>
  
          <section className="new-project-card">
            <div className="new-project-section-heading">
              <div>
                <span className="new-project-step">
                  02
                </span>
  
                <h2>上传小说文件</h2>
              </div>
  
              {fileItems.length > 0 && (
                <button
                  type="button"
                  className="new-project-text-button danger"
                  disabled={isUploading}
                  onClick={clearFiles}
                >
                  清空全部
                </button>
              )}
            </div>
  
            <label className="new-project-upload-area">
              <input
                type="file"
                accept=".txt,text/plain"
                multiple
                disabled={isUploading || isReadingFiles}
                onChange={handleFileInput}
              />
  
              <strong>
                {isReadingFiles
                  ? "正在读取并校验文件……"
                  : "选择一个或多个 TXT 文件"}
              </strong>
  
              <span>
                支持多次添加；单文件最大 10 MB；
                总大小最大 20 MB；最多 50 个文件
              </span>
            </label>
  
            {validationMessages.length > 0 && (
              <div
                className="new-project-message warning"
                role="status"
              >
                {validationMessages.map((message) => (
                  <p key={message}>
                    {message}
                  </p>
                ))}
              </div>
            )}
  
            {fileItems.length > 0 && (
              <div className="new-project-stat-grid">
                <div>
                  <span>文件数量</span>
                  <strong>{fileItems.length}</strong>
                </div>
  
                <div>
                  <span>总大小</span>
  
                  <strong>
                    {formatBytes(projectSummary.sizeBytes)}
                  </strong>
                </div>
  
                <div>
                  <span>总字符数</span>
  
                  <strong>
                    {projectSummary.characterCount.toLocaleString()}
                  </strong>
                </div>
  
                <div>
                  <span>总行数</span>
  
                  <strong>
                    {projectSummary.lineCount.toLocaleString()}
                  </strong>
                </div>
              </div>
            )}
          </section>
  
          {fileItems.length > 0 && (
            <section className="new-project-file-workspace">
              <div className="new-project-file-panel">
                <div className="new-project-panel-heading">
                  <h2>文件顺序</h2>
                  <span>后端将按照此顺序处理</span>
                </div>
  
                <div className="new-project-file-list">
                  {fileItems.map((item, index) => (
                    <article
                      key={item.id}
                      className={
                        item.id === selectedFileId
                          ? "new-project-file-item selected"
                          : "new-project-file-item"
                      }
                    >
                      <button
                        type="button"
                        className="new-project-file-main"
                        onClick={() => {
                          setSelectedFileId(item.id);
                        }}
                      >
                        <span className="new-project-file-order">
                          {index + 1}
                        </span>
  
                        <span className="new-project-file-information">
                          <strong>
                            {item.file.name}
                          </strong>
  
                          <small>
                            {formatBytes(item.file.size)}
                            {" · "}
                            {item.characterCount.toLocaleString()}
                            字符
                          </small>
                        </span>
                      </button>
  
                      <div className="new-project-file-actions">
                        <button
                          type="button"
                          title="上移"
                          disabled={index === 0 || isUploading}
                          onClick={() => moveFile(index, -1)}
                        >
                          ↑
                        </button>
  
                        <button
                          type="button"
                          title="下移"
                          disabled={
                            index === fileItems.length - 1
                            || isUploading
                          }
                          onClick={() => moveFile(index, 1)}
                        >
                          ↓
                        </button>
  
                        <button
                          type="button"
                          title="删除"
                          className="danger"
                          disabled={isUploading}
                          onClick={() => removeFile(item.id)}
                        >
                          ×
                        </button>
                      </div>
                    </article>
                  ))}
                </div>
              </div>
  
              <div className="new-project-preview-panel">
                <div className="new-project-panel-heading">
                  <h2>文本预览</h2>
  
                  {selectedFileItem && (
                    <span>
                      {selectedFileItem.lineCount.toLocaleString()}
                      行
                    </span>
                  )}
                </div>
  
                {selectedFileItem ? (
                  <>
                    <div className="new-project-preview-meta">
                      <strong>
                        {selectedFileItem.file.name}
                      </strong>
  
                      <span>
                        {selectedFileItem.characterCount.toLocaleString()}
                        字符
                      </span>
                    </div>
  
                    <pre className="new-project-preview-text">
                      {selectedFileItem.text.slice(0, 5000)}
  
                      {selectedFileItem.text.length > 5000
                        ? "\n\n……预览仅显示前 5000 个字符"
                        : ""}
                    </pre>
                  </>
                ) : (
                  <div className="new-project-empty">
                    请选择一个文件查看预览
                  </div>
                )}
              </div>
            </section>
          )}
  
          {uploadError && (
            <div
              className="new-project-message error"
              role="alert"
            >
              {uploadError}
            </div>
          )}
  
          <footer className="new-project-submit-area">
            <div>
              <strong>上传后将自动保存项目</strong>
  
              <span>
                上传成功后会进入项目工作区。
              </span>
            </div>
  
            <button
              type="submit"
              className="new-project-submit-button"
              disabled={
                isUploading
                || isReadingFiles
                || fileItems.length === 0
              }
            >
              {isUploading
                ? "正在上传并创建项目……"
                : "上传并创建项目"}
            </button>
          </footer>
        </form>
      </main>
    );
  }
  
  
  export default NewProjectPage;