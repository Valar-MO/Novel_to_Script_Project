import { useNavigate } from "react-router-dom";


function HomePage() {
  const navigate = useNavigate();

  return (
    <div className="app">
      <header className="navbar">
        <h1 className="logo">Novel2Script</h1>
        <span className="subtitle">AI 小说转剧本工具</span>
      </header>

      <main className="main">
        <section className="hero">
          <p className="tag">NOVEL TO SCREENPLAY</p>

          <h2>让小说改编成结构化影视剧本</h2>

          <p className="description">
            上传或输入小说文本，系统将完成人物提取、场景划分、
            场景卡生成和影视剧本改写。
          </p>

          <div className="hero-actions">
            <button
              type="button"
              className="start-button"
              onClick={() => navigate("/project/new")}
            >
              创建新项目
            </button>

            <button
              type="button"
              className="projects-button"
              onClick={() => navigate("/projects")}
            >
              查看已有项目
            </button>
          </div>
        </section>

        <section className="workflow">
          <h3>处理流程</h3>

          <div className="workflow-list">
            <div className="workflow-item">
              <span>01</span>
              <p>输入小说</p>
            </div>

            <div className="workflow-item">
              <span>02</span>
              <p>分析故事</p>
            </div>

            <div className="workflow-item">
              <span>03</span>
              <p>划分场景</p>
            </div>

            <div className="workflow-item">
              <span>04</span>
              <p>生成剧本</p>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}


export default HomePage;
