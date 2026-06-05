
import { useRef, useState } from "react";
import { useNavigate } from "react-router";

const MAX_FILE_SIZE = 10 * 1024 * 1024;
const MAX_TOTAL_SIZE = 20 * 1024 * 1024;
const MAX_FILE_COUNT = 50;
const PREVIEW_LENGTH = 2000;

function formatFileSize(size) {
  if (size < 1024) {
    return `${size} B`;
  }

  if (size < 1024 * 1024) {
    return `${(size / 1024).toFixed(1)} KB`;
  }

  return `${(size / 1024 / 1024).toFixed(2)} MB`;
}

function compareFileNames(firstFile, secondFile) {
  return firstFile.name.localeCompare(secondFile.name, "zh-CN", {
    numeric: true,
    sensitivity: "base",
  });
}

function createFileId(file) {
  return `${file.name}-${file.size}-${file.lastModified}`;
}

function NewProjectPage() {
  const navigate = useNavigate();
  const fileInputRef = useRef(null);

  const [projectName, setProjectName] = useState("");
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [activePreviewId, setActivePreviewId] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [successMessage, setSuccessMessage] = useState("");

  const totalFileSize = selectedFiles.reduce(
    (total, fileItem) => total + fileItem.size,
    0,
  );

  const totalCharacterCount = selectedFiles.reduce(
    (total, fileItem) => total + fileItem.text.length,
    0,
  );

  const totalLineCount = selectedFiles.reduce(
    (total, fileItem) => total + fileItem.lineCount,
    0,
  );

  const activePreviewFile =
    selectedFiles.find(
      (fileItem) => fileItem.id === activePreviewId,
    ) ?? selectedFiles[0];

  const resetFileInput = () => {
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleFileChange = async (event) => {
    const chosenFiles = Array.from(event.target.files ?? []);

    setErrorMessage("");
    setSuccessMessage("");

    if (chosenFiles.length === 0) {
      return;
    }

    const errors = [];
    const existingFileIds = new Set(
      selectedFiles.map((fileItem) => fileItem.id),
    );

    let candidateFiles = [];

    for (const file of chosenFiles) {
      const fileId = createFileId(file);

      if (existingFileIds.has(fileId)) {
        errors.push(`“${file.name}”已经添加，请勿重复上传。`);
        continue;
      }

      if (!file.name.toLowerCase().endsWith(".txt")) {
        errors.push(`“${file.name}”不是 TXT 文件，已跳过。`);
        continue;
      }

      if (file.size === 0) {
        errors.push(`“${file.name}”是空文件，已跳过。`);
        continue;
      }

      if (file.size > MAX_FILE_SIZE) {
        errors.push(
          `“${file.name}”超过单文件 10 MB 限制，已跳过。`,
        );
        continue;
      }

      candidateFiles.push(file);
      existingFileIds.add(fileId);
    }

    const remainingFileSlots =
      MAX_FILE_COUNT - selectedFiles.length;

    if (candidateFiles.length > remainingFileSlots) {
      errors.push(
        `每个项目最多上传 ${MAX_FILE_COUNT} 个文件，超出部分已跳过。`,
      );

      candidateFiles = candidateFiles.slice(
        0,
        remainingFileSlots,
      );
    }

    candidateFiles.sort(compareFileNames);

    const filesWithinTotalSize = [];
    let accumulatedSize = totalFileSize;

    for (const file of candidateFiles) {
      if (accumulatedSize + file.size > MAX_TOTAL_SIZE) {
        errors.push(
          `加入“${file.name}”后会超过总文件 20 MB 限制，已跳过。`,
        );
        continue;
      }

      filesWithinTotalSize.push(file);
      accumulatedSize += file.size;
    }

    const readResults = await Promise.all(
      filesWithinTotalSize.map(async (file) => {
        try {
          const text = await file.text();

          if (!text.trim()) {
            return {
              error: `“${file.name}”没有有效文本内容，已跳过。`,
              fileItem: null,
            };
          }

          const replacementCharacterCount = (
            text.match(/\uFFFD/g) || []
          ).length;

          if (replacementCharacterCount > 5) {
            return {
              error:
                `“${file.name}”可能不是 UTF-8 编码，` +
                "读取后出现乱码，已跳过。",
              fileItem: null,
            };
          }

          return {
            error: null,
            fileItem: {
              id: createFileId(file),
              name: file.name,
              size: file.size,
              text,
              lineCount: text.split(/\r?\n/).length,
              lastModified: file.lastModified,
              originalFile: file,
            },
          };
        } catch (error) {
          console.error(`读取 ${file.name} 失败：`, error);

          return {
            error: `“${file.name}”读取失败，已跳过。`,
            fileItem: null,
          };
        }
      }),
    );

    const newFileItems = [];
    const readErrors = [];

    for (const result of readResults) {
      if (result.fileItem) {
        newFileItems.push(result.fileItem);
      }

      if (result.error) {
        readErrors.push(result.error);
      }
    }

    const allErrors = [...errors, ...readErrors];

    if (newFileItems.length > 0) {
      const nextFiles = [...selectedFiles, ...newFileItems];

      setSelectedFiles(nextFiles);

      if (!activePreviewId) {
        setActivePreviewId(nextFiles[0].id);
      }

      setSuccessMessage(
        `成功添加 ${newFileItems.length} 个 TXT 文件。` +
          "请检查文件顺序，系统会按当前顺序处理文本。",
      );
    }

    if (allErrors.length > 0) {
      setErrorMessage(allErrors.join("\n"));
    }

    resetFileInput();
  };

  const removeFile = (fileId) => {
    const removedIndex = selectedFiles.findIndex(
      (fileItem) => fileItem.id === fileId,
    );

    const remainingFiles = selectedFiles.filter(
      (fileItem) => fileItem.id !== fileId,
    );

    setSelectedFiles(remainingFiles);
    setErrorMessage("");
    setSuccessMessage("");

    if (fileId === activePreviewId) {
      const nextPreviewIndex = Math.min(
        Math.max(removedIndex, 0),
        remainingFiles.length - 1,
      );

      setActivePreviewId(
        remainingFiles[nextPreviewIndex]?.id ?? null,
      );
    }
  };

  const clearAllFiles = () => {
    setSelectedFiles([]);
    setActivePreviewId(null);
    setErrorMessage("");
    setSuccessMessage("");
    resetFileInput();
  };

  const moveFile = (currentIndex, direction) => {
    const targetIndex = currentIndex + direction;

    if (
      targetIndex < 0 ||
      targetIndex >= selectedFiles.length
    ) {
      return;
    }

    const reorderedFiles = [...selectedFiles];

    [
      reorderedFiles[currentIndex],
      reorderedFiles[targetIndex],
    ] = [
      reorderedFiles[targetIndex],
      reorderedFiles[currentIndex],
    ];

    setSelectedFiles(reorderedFiles);
  };

  const sortFilesByName = () => {
    const sortedFiles = [...selectedFiles].sort(
      compareFileNames,
    );

    setSelectedFiles(sortedFiles);
    setSuccessMessage("文件已按文件名重新排序。");
    setErrorMessage("");
  };

  const handleSubmit = (event) => {
    event.preventDefault();

    setErrorMessage("");
    setSuccessMessage("");

    if (!projectName.trim()) {
      setErrorMessage("请输入项目名称。");
      return;
    }

    if (selectedFiles.length === 0) {
      setErrorMessage("请至少上传一个小说 TXT 文件。");
      return;
    }

    setSuccessMessage(
      `前端校验已通过，共 ${selectedFiles.length} 个文件。` +
        "下一阶段将按当前文件顺序提交给后端进行长文本分析。",
    );
  };

  const previewText = activePreviewFile
    ? activePreviewFile.text.length > PREVIEW_LENGTH
      ? `${activePreviewFile.text.slice(
          0,
          PREVIEW_LENGTH,
        )}\n\n……文本预览结束，完整内容将在后端处理。`
      : activePreviewFile.text
    : "";

  return (
    <div className="app">
      <header className="navbar">
        <h1 className="logo">Novel2Script</h1>

        <button
          type="button"
          className="back-button"
          onClick={() => navigate("/")}
        >
          返回首页
        </button>
      </header>

      <main className="project-page">
        <section className="project-header">
          <p className="tag">NEW PROJECT</p>

          <h2>创建小说改编项目</h2>

          <p>
            上传一个或多个 TXT 小说文件。多个文件应属于同一部小说，
            系统后续将按照当前文件顺序进行长文本处理、人物提取、
            情节分析、场景划分和剧本生成。
          </p>
        </section>

        <form
          className="project-form"
          onSubmit={handleSubmit}
        >
          <div className="form-group">
            <label htmlFor="project-name">
              项目名称
              <span className="required-mark">*</span>
            </label>

            <input
              id="project-name"
              type="text"
              value={projectName}
              onChange={(event) =>
                setProjectName(event.target.value)
              }
              placeholder="例如：《短篇小说》剧本改编"
              maxLength={100}
            />

            <span className="field-tip">
              用于区分和管理不同的小说改编项目。
            </span>
          </div>

          <div className="form-group">
            <label>
              小说文件
              <span className="required-mark">*</span>
            </label>

            <div className="upload-area">
              <input
                ref={fileInputRef}
                id="novel-files"
                className="file-input"
                type="file"
                accept=".txt,text/plain"
                multiple
                onChange={handleFileChange}
              />

              <label
                className="upload-label"
                htmlFor="novel-files"
              >
                <span className="upload-icon">＋</span>

                <strong>
                  {selectedFiles.length > 0
                    ? "继续添加 TXT 文件"
                    : "选择一个或多个 TXT 文件"}
                </strong>

                <span>
                  单个文件不超过 10 MB，总大小不超过 20 MB，
                  最多上传 50 个文件
                </span>
              </label>
            </div>

            <span className="field-tip">
              多个文件应属于同一部小说。首次上传时会按文件名自然排序，
              之后可以手动调整处理顺序。
            </span>
          </div>

          {errorMessage && (
            <div
              className="message message-error"
              role="alert"
            >
              {errorMessage}
            </div>
          )}

          {successMessage && (
            <div className="message message-success">
              {successMessage}
            </div>
          )}

          {selectedFiles.length > 0 && (
            <section className="files-section">
              <div className="files-section-header">
                <div>
                  <h3>已上传文件</h3>
                  <p>
                    系统将从上到下按顺序处理这些文件。
                  </p>
                </div>

                <div className="files-header-actions">
                  <button
                    type="button"
                    className="text-action-button"
                    onClick={sortFilesByName}
                    disabled={selectedFiles.length < 2}
                  >
                    按文件名排序
                  </button>

                  <button
                    type="button"
                    className="text-action-button danger-action"
                    onClick={clearAllFiles}
                  >
                    清空全部
                  </button>
                </div>
              </div>

              <div className="project-statistics">
                <div className="statistic-item">
                  <span>文件数量</span>
                  <strong>{selectedFiles.length}</strong>
                </div>

                <div className="statistic-item">
                  <span>总文件大小</span>
                  <strong>
                    {formatFileSize(totalFileSize)}
                  </strong>
                </div>

                <div className="statistic-item">
                  <span>总字符数</span>
                  <strong>
                    {totalCharacterCount.toLocaleString()}
                  </strong>
                </div>

                <div className="statistic-item">
                  <span>总文本行数</span>
                  <strong>
                    {totalLineCount.toLocaleString()}
                  </strong>
                </div>
              </div>

              <div className="file-list">
                {selectedFiles.map((fileItem, index) => (
                  <div
                    key={fileItem.id}
                    className={
                      fileItem.id === activePreviewFile?.id
                        ? "file-list-item file-list-item-active"
                        : "file-list-item"
                    }
                  >
                    <div className="file-order">
                      {index + 1}
                    </div>

                    <button
                      type="button"
                      className="file-information"
                      onClick={() =>
                        setActivePreviewId(fileItem.id)
                      }
                    >
                      <span className="file-name">
                        {fileItem.name}
                      </span>

                      <span className="file-metadata">
                        {formatFileSize(fileItem.size)}
                        {" · "}
                        {fileItem.text.length.toLocaleString()}
                        个字符
                      </span>
                    </button>

                    <div className="file-item-actions">
                      <button
                        type="button"
                        className="order-button"
                        onClick={() => moveFile(index, -1)}
                        disabled={index === 0}
                        aria-label={`上移 ${fileItem.name}`}
                        title="上移"
                      >
                        ↑
                      </button>

                      <button
                        type="button"
                        className="order-button"
                        onClick={() => moveFile(index, 1)}
                        disabled={
                          index === selectedFiles.length - 1
                        }
                        aria-label={`下移 ${fileItem.name}`}
                        title="下移"
                      >
                        ↓
                      </button>

                      <button
                        type="button"
                        className="delete-file-button"
                        onClick={() =>
                          removeFile(fileItem.id)
                        }
                      >
                        删除
                      </button>
                    </div>
                  </div>
                ))}
              </div>

              {activePreviewFile && (
                <div className="preview-section">
                  <div className="preview-header">
                    <div>
                      <h3>文本预览</h3>
                      <p>{activePreviewFile.name}</p>
                    </div>

                    <span>
                      最多显示前{" "}
                      {PREVIEW_LENGTH.toLocaleString()} 个字符
                    </span>
                  </div>

                  <pre className="text-preview">
                    {previewText}
                  </pre>
                </div>
              )}
            </section>
          )}

          <div className="form-actions">
            <button
              type="button"
              className="secondary-button"
              onClick={() => navigate("/")}
            >
              取消
            </button>

            <button
              type="submit"
              className="start-button submit-button"
              disabled={
                !projectName.trim() ||
                selectedFiles.length === 0
              }
            >
              开始分析
            </button>
          </div>
        </form>
      </main>
    </div>
  );
}

export default NewProjectPage;

