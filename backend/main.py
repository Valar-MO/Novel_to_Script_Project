from fastapi import FastAPI

app = FastAPI(
    title="Novel2Script API",
    description="Novel2Script 小说转剧本系统后端接口",
    version="0.1.0",
)


@app.get("/api/health", tags=["system"])
def health_check() -> dict[str, str]:
    """检查后端服务是否正常运行。"""
    return {
        "status": "ok",
        "service": "Novel2Script API",
    }