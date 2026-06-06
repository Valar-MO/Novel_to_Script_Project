
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.narrative_analysis import router as narrative_analysis_router
from backend.api.projects import router as projects_router


app = FastAPI(
    title="Novel2Script API",
    description="Novel2Script 小说转剧本系统后端接口",
    version="0.1.0",
)

# 本地开发环境允许访问后端的前端地址。
# localhost 和 127.0.0.1 会被浏览器视为不同来源，因此分别配置。
allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(projects_router)
app.include_router(narrative_analysis_router)


@app.get("/api/health", tags=["system"])
def health_check() -> dict[str, str]:
    """检查后端服务是否正常运行。"""

    return {
        "status": "ok",
        "service": "Novel2Script API",
    }
