# Novel2Script

Novel2Script 是一个本地运行的小说转剧本工具。当前版本主要用于演示从小说文本到可编辑剧本场景的基础流程。

目前支持：

1. 上传 `.txt` 小说文件；
2. 自动切分并浏览章节；
3. 使用云端大模型 API 抽取人物锚点和人物关系；
4. 生成项目级人物关系，并支持人工编辑；
5. 根据小说文本生成可编辑的剧本场景；
6. 对单个剧本场景进行编辑或重新生成。

当前默认使用 DeepSeek API，不再依赖本地 Ollama。

## 环境要求

- Python 3.11 或更高版本
- Node.js 20 或更高版本
- DeepSeek API Key

## 后端启动

安装后端依赖：

```powershell
pip install -r backend/requirements.txt
```

复制环境变量模板：

```powershell
copy .env.example .env
```

编辑 `.env`，填写自己的 DeepSeek API Key：

```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_real_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
```

启动后端：

```powershell
python -m uvicorn backend.main:app --reload
```

后端默认地址：

```text
http://127.0.0.1:8000
```

## 前端启动

进入前端目录并安装依赖：

```powershell
cd frontend
npm install
```

如需修改后端地址，可以复制前端环境变量模板：

```powershell
copy .env.example .env
```

启动前端：

```powershell
npm run dev
```

浏览器打开终端中显示的 Vite 地址，通常是：

```text
http://127.0.0.1:5173
```

## 基本演示流程

1. 创建新项目；
2. 上传一个或多个 `.txt` 小说文件；
3. 进入项目工作区；
4. 点击 AI 叙事分析；
5. 点击分析人物关系；
6. 查看、固定或编辑核心人物关系；
7. 点击生成剧本；
8. 查看生成的剧本场景；
9. 按需编辑场景或单场重新生成。

## 常用命令

运行后端测试：

```powershell
python -m unittest discover backend/tests
```

构建前端：

```powershell
cd frontend
npm run build
```

测试真实 API 的人物锚点抽取：

```powershell
python scripts/test_mention_extraction_real.py
```

测试真实 API 的人物关系抽取：

```powershell
python scripts/test_relation_extraction_real.py
```

## 环境变量说明

后端主要环境变量：

```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_real_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
LLM_TEMPERATURE=0
LLM_API_TIMEOUT_SECONDS=180
LLM_API_THINKING_ENABLED=false
LLM_API_REASONING_EFFORT=
```

也支持 OpenAI-compatible API 通用配置：

```env
LLM_API_KEY=
LLM_API_BASE_URL=
LLM_API_MODEL=
```

前端环境变量：

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## 当前功能边界

当前版本适合作为本地 demo 和开发原型使用，暂不作为生产服务。

已经实现的核心能力：

- 小说文件上传与章节浏览；
- 后台 AI 分析任务；
- 人物锚点抽取；
- 自由人物关系抽取；
- 项目级人物关系构建；
- 核心人物固定；
- 人物关系新增、编辑、删除；
- 基于小说文本生成剧本场景；
- 剧本场景编辑和单场重新生成。

暂未重点实现：

- 复杂关系图布局；
- 关系时间线；
- 关系历史版本管理；
- 自动检测剧本是否违反人物关系；
- 生产级用户系统和权限管理。

## 注意事项

- 本地生成的数据保存在 `data/` 目录下，并已被 Git 忽略；
- `.env` 文件已被 Git 忽略，不要提交真实 API Key；
- 如果后端启动时报 API Key 缺失，请检查 `.env` 中的 `DEEPSEEK_API_KEY`；
- 如果前端无法连接后端，请确认后端正在 `http://127.0.0.1:8000` 运行。
