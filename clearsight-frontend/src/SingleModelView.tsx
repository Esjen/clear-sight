import { useState, FormEvent } from "react";

type ModelName = "random_forest" | "xgboost" | "lightgbm";

interface TopFeature {
  feature: string;
  shap_value: number;
  direction: string;
}

interface PredictionResult {
  model: ModelName;
  prediction: "malicious" | "benign";
  confidence: number;
  probability_malicious: number;
  top_features: TopFeature[];
  true_label?: "malicious" | "benign";
}

const API_URL = "http://localhost:3001";

const MODEL_LABELS: Record<ModelName, string> = {
  random_forest: "Random Forest",
  xgboost: "XGBoost",
  lightgbm: "LightGBM",
};

function SingleModelView() {
  const [rowIndex, setRowIndex] = useState("0");
  const [model, setModel] = useState<ModelName>("lightgbm");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ row: Number(rowIndex), model }),
      });

      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.error || `Request failed (${response.status})`);
      }

      const data: PredictionResult = await response.json();
      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not reach the ClearSight backend. Is it running on port 3001?"
      );
    } finally {
      setLoading(false);
    }
  };

  const maxAbsShap = result
    ? Math.max(...result.top_features.map((f) => Math.abs(f.shap_value)))
    : 1;

  return (
    <div className="layout">
        <section className="panel panel-controls">
          <h2>Run a detection</h2>
          <p className="panel-hint">
            Select a file from the EMBER 2018 test set by its row index, and
            choose which trained model should evaluate it.
          </p>

          <form onSubmit={handleSubmit}>
            <label htmlFor="row-index">Test file index</label>
            <input
              id="row-index"
              type="number"
              min={0}
              max={199999}
              value={rowIndex}
              onChange={(e) => setRowIndex(e.target.value)}
              required
            />
            <span className="field-note">0 to 199,999</span>

            <label htmlFor="model-select">Model</label>
            <select
              id="model-select"
              value={model}
              onChange={(e) => setModel(e.target.value as ModelName)}
            >
              <option value="random_forest">Random Forest</option>
              <option value="xgboost">XGBoost</option>
              <option value="lightgbm">LightGBM</option>
            </select>

            <button type="submit" disabled={loading}>
              {loading ? "Analyzing file..." : "Run detection"}
            </button>
          </form>
        </section>

        <section className="panel panel-result">
          {!result && !loading && !error && (
            <div className="empty-state">
              <p>
                Choose a file and a model, then run a detection to see the
                prediction and the evidence behind it.
              </p>
            </div>
          )}

          {loading && (
            <div className="empty-state">
              <p>Extracting features and computing the explanation…</p>
            </div>
          )}

          {error && (
            <div className="error-state">
              <p className="error-title">Detection failed</p>
              <p>{error}</p>
            </div>
          )}

          {result && (
            <>
              <div
                className={`verdict verdict-${result.prediction}`}
                role="status"
              >
                <span className="verdict-label">
                  {result.prediction === "malicious" ? "Malicious" : "Benign"}
                </span>
                <span className="verdict-confidence">
                  {(result.confidence * 100).toFixed(1)}% confidence
                </span>
                <span className="verdict-meta">
                  Model: {MODEL_LABELS[result.model]}
                  {result.true_label && (
                    <>
                      {" · "}
                      Reference label: {result.true_label}
                      {result.true_label === result.prediction
                        ? " (prediction matches)"
                        : " (prediction differs)"}
                    </>
                  )}
                </span>
              </div>

              <div className="explanation">
                <h3>What drove this decision</h3>
                <p className="panel-hint">
                  The features below had the largest effect on this
                  prediction, ranked by strength. Red bars push the
                  prediction toward malicious; green bars push it toward
                  benign. Longer bars had a stronger effect.
                </p>

                <ul className="feature-list">
                  {result.top_features.map((f) => {
                    const widthPct =
                      (Math.abs(f.shap_value) / maxAbsShap) * 100;
                    const isMalicious = f.shap_value > 0;
                    return (
                      <li key={f.feature} className="feature-row">
                        <span className="feature-name">{f.feature}</span>
                        <div className="feature-bar-track">
                          <div
                            className={`feature-bar ${
                              isMalicious ? "feature-bar-malicious" : "feature-bar-benign"
                            }`}
                            style={{ width: `${widthPct}%` }}
                          />
                        </div>
                        <span className="feature-value">
                          {f.shap_value > 0 ? "+" : ""}
                          {f.shap_value.toFixed(3)}
                        </span>
                      </li>
                    );
                  })}
                </ul>
              </div>
            </>
          )}
        </section>
      </div>
  );
}

export default SingleModelView;
