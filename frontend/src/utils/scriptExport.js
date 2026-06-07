function getCharacterName(character) {
  if (typeof character === "string") {
    return character.trim();
  }

  if (
    character
    && typeof character === "object"
  ) {
    return String(
      character.name
      || character.canonical_name
      || "",
    ).trim();
  }

  return "";
}


function getSceneCharacters(scene) {
  return (scene.characters || [])
    .map(getCharacterName)
    .filter(Boolean);
}


function getSceneTitle(scene, index) {
  const sceneNumber = (
    scene.scene_number
    || index + 1
  );

  const heading = (
    scene.heading
    || [
      scene.interior_exterior,
      scene.location,
      scene.time_of_day,
    ]
      .filter(Boolean)
      .join("·")
    || "未命名场景"
  );

  return {
    sceneNumber,
    heading,
  };
}


export function buildScriptText(
  projectName,
  scenes,
) {
  const lines = [
    projectName || "未命名剧本",
    "=".repeat(32),
    "",
  ];

  scenes.forEach((scene, index) => {
    const {
      sceneNumber,
      heading,
    } = getSceneTitle(scene, index);

    const characters = getSceneCharacters(scene);

    lines.push(`第 ${sceneNumber} 场`);
    lines.push(heading);

    if (characters.length > 0) {
      lines.push(`人物：${characters.join("、")}`);
    }

    lines.push("");
    lines.push(scene.script_text || "");
    lines.push("");
    lines.push("-".repeat(32));
    lines.push("");
  });

  return lines.join("\n");
}


export function buildScriptMarkdown(
  projectName,
  scenes,
) {
  const lines = [
    `# ${projectName || "未命名剧本"}`,
    "",
  ];

  scenes.forEach((scene, index) => {
    const {
      sceneNumber,
      heading,
    } = getSceneTitle(scene, index);

    const characters = getSceneCharacters(scene);

    lines.push(
      `## 第 ${sceneNumber} 场 ${heading}`,
    );
    lines.push("");

    if (characters.length > 0) {
      lines.push(
        `**人物：** ${characters.join("、")}`,
      );
      lines.push("");
    }

    lines.push(scene.script_text || "");
    lines.push("");
    lines.push("---");
    lines.push("");
  });

  return lines.join("\n");
}


export function sanitizeFileName(fileName) {
  const sanitized = String(fileName || "剧本")
    .trim()
    .replace(/[\\/:*?"<>|]/g, "_");

  return sanitized || "剧本";
}


export function downloadTextFile(
  fileName,
  content,
  mimeType,
) {
  const blob = new Blob(
    ["\ufeff", content],
    {
      type: `${mimeType};charset=utf-8`,
    },
  );

  const objectUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");

  link.href = objectUrl;
  link.download = fileName;

  document.body.appendChild(link);
  link.click();
  link.remove();

  URL.revokeObjectURL(objectUrl);
}
