import { useState } from "react";
import "./App.css";
import SingleModelView from "./SingleModelView";
import CompareView from "./CompareView";

type Tab = "single" | "compare";

function App() {
  const [tab, setTab] = useState<Tab>("single");

  return (
    <div className="page">
      <header className="page-header">
        <h1>ClearSight</h1>
        <p>Explainable static malware detection for security operations</p>
      </header>

      <div className="tabs" role="tablist" aria-label="ClearSight views">
        <button
          role="tab"
          aria-selected={tab === "single"}
          className={tab === "single" ? "tab tab-active" : "tab"}
          onClick={() => setTab("single")}
        >
          Single model
        </button>
        <button
          role="tab"
          aria-selected={tab === "compare"}
          className={tab === "compare" ? "tab tab-active" : "tab"}
          onClick={() => setTab("compare")}
        >
          Compare models
        </button>
      </div>

      {/* Both views stay mounted so switching tabs does not lose your results. */}
      <div hidden={tab !== "single"}>
        <SingleModelView />
      </div>
      <div hidden={tab !== "compare"}>
        <CompareView />
      </div>
    </div>
  );
}

export default App;
