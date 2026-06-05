import { Route, Routes } from "react-router";
import "./App.css";

import HomePage from "./pages/HomePage";
import NewProjectPage from "./pages/NewProjectPage";

function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/project/new" element={<NewProjectPage />} />
    </Routes>
  );
}

export default App;