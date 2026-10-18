import { useEffect, useState, FormEvent } from "react";

type ModelKey = "random_forest" | "xgboost" | "lightgbm";

interface CompareFeature {
  rank: number;
  feature: string;
  shap_value: number;
  direction: string;
  shared_with: ModelKey[];
}

interface CompareModel {
  probability_malicious: number;
  prediction: "malicious" | "benign";
  top_features: CompareFeature[];
}

interface CompareResult {
  row: number;
  true_label: "malicious" | "benign";
  top_k_for_jaccard: number;
  models: Record<ModelKey, CompareModel>;
  pairwise_jaccard: {
    xgboost_vs_lightgbm: number;
    random_forest_vs_xgboost: number;
    random_forest_vs_lightgbm: number;
  };
}

interface AgreementPair {
  pair: string;
  shared: number;
  jaccard: number;
  ci_low: number;
  ci_high: number;
}

interface Agreement {
  n_files: number;
  n_bootstrap: number;
  primary_k: number;
  scope: string;
  pairs: AgreementPair[];
  h2: {
    fraction_boosting_pair_higher_than_both_rf_pairs: number;
    gap_boosting_minus_mean_rf_pairs: number;
    gap_ci_low: number;
    gap_ci_high: number;
  };
}

const API_URL = "http://localhost:3001";
const MODEL_ORDER: ModelKey[] = ["random_forest", "xgboost", "lightgbm"];
const MODEL_LABELS: Record<ModelKey, string> = {
  random_forest: "Random Forest",
  xgboost: "XGBoost",
  lightgbm: "LightGBM",
};

async function readError(response: Response): Promise<string> {
  const body = await response.json().catch(() => ({}));
  return body.message || body.error || `Request failed (${response.status})`;
}

function CompareView() {
  const [agreement, setAgreement] = useState<Agreement | null>(null);
  const [agreementError, setAgreementError] = useState<string | null>(null);

  const [rowInput, setRowInput] = useState("");
  const [result, setResult] = useState<CompareResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_URL}/agreement`)
      .then(async (r) => {
        if (!r.ok) throw new Error(await readError(r));
        setAgreement(await r.json());
      })
      .catch((e) =>
        setAgreementError(
          e instanceof Error ? e.message : "Could not reach the ClearSight backend."
        )
      );
  }, []);

  const runCompare = async (query: string) => {
    setLoading(true);
    setError(null);
    try {
      const r = await fetch(`${API_URL}/compare?${query}`);
      if (!r.ok) throw new Error(await readError(r));
      const data: CompareResult = await r.json();
      setResult(data);
      setRowInput(String(data.row));
    } catch (e) {
      setResult(null);
      setError(e instanceof Error ? e.message : "Could not reach the ClearSight backend.");
    } finally {
      setLoading(false);
    }
  };

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    runCompare(`row=${encodeURIComponent(rowInput)}`);
  };

  return (
    <div className="compare">
      <section className="panel">
        <h2>Do the models agree?</h2>
        {agreementError && (
          <p className="panel-hint">Overall agreement is not available yet: {agreementError}</p>
        )}
        {!agreement && !agreementError && <p className="panel-hint">Loading…</p>}
        {agreement && (
          <>
            <p className="panel-hint">
              Overlap between the top {agreement.primary_k} features each pair of models relies on,
              measured on {agreement.n_files.toLocaleString()} test files. The thin line shows a 95%
              interval from {agreement.n_bootstrap.toLocaleString()} bootstrap resamples. A value of 1
              means identical feature lists and 0 means none in common.
            </p>
            <ul className="agree-list">
              {agreement.pairs.map((p) => (
                <li key={p.pair} className="agree-row">
                  <span className="agree-name">{p.pair}</span>
                  <div className="agree-track">
                    <div className="agree-fill" style={{ width: `${p.jaccard * 100}%` }} />
                    <div
                      className="agree-ci"
                      style={{
                        left: `${p.ci_low * 100}%`,
                        width: `${Math.max((p.ci_high - p.ci_low) * 100, 0.4)}%`,
                      }}
                    />
                  </div>
                  <span className="agree-value">
                    {p.jaccard.toFixed(3)}
                    <span className="agree-range">
                      {" "}
                      ({p.ci_low.toFixed(3)} to {p.ci_high.toFixed(3)})
                    </span>
                  </span>
                </li>
              ))}
            </ul>
            <p className="note">
              In {(agreement.h2.fraction_boosting_pair_higher_than_both_rf_pairs * 100).toFixed(1)}% of
              resamples, XGBoost and LightGBM agreed more with each other than either did with Random
              Forest. {agreement.scope}
            </p>
          </>
        )}
      </section>

      <section className="panel">
        <h2>Compare one file</h2>
        <p className="panel-hint">
          See what each model relied on for the same file. Explanations are available for the
          1,000 test files in the explanation sample.
        </p>
        <form className="compare-form" onSubmit={onSubmit}>
          <label htmlFor="cmp-row">Test file index</label>
          <input
            id="cmp-row"
            type="number"
            min={0}
            max={199999}
            value={rowInput}
            onChange={(e) => setRowInput(e.target.value)}
            required
          />
          <button type="submit" disabled={loading}>
            {loading ? "Comparing…" : "Compare models"}
          </button>
          <button
            type="button"
            className="button-secondary"
            disabled={loading}
            onClick={() => runCompare("random=1")}
          >
            Random sampled file
          </button>
        </form>

        {error && (
          <div className="error-state compact">
            <p className="error-title">Comparison failed</p>
            <p>{error}</p>
          </div>
        )}

        {result && (
          <div className="compare-result">
            <p className="compare-meta">
              File {result.row}. Reference label: <strong>{result.true_label}</strong>. Overlap of
              the top {result.top_k_for_jaccard} features for this file:{" "}
              XGBoost and LightGBM {result.pairwise_jaccard.xgboost_vs_lightgbm.toFixed(2)}, Random
              Forest and XGBoost {result.pairwise_jaccard.random_forest_vs_xgboost.toFixed(2)},
              Random Forest and LightGBM{" "}
              {result.pairwise_jaccard.random_forest_vs_lightgbm.toFixed(2)}.
            </p>

            <div className="cmp-grid">
              {MODEL_ORDER.map((m) => {
                const model = result.models[m];
                const maxAbs = Math.max(...model.top_features.map((f) => Math.abs(f.shap_value)), 1e-9);
                const correct = model.prediction === result.true_label;
                return (
                  <div key={m} className="cmp-col">
                    <h3>{MODEL_LABELS[m]}</h3>
                    <div className={`cmp-verdict verdict-${model.prediction}`}>
                      <strong>{model.prediction === "malicious" ? "Malicious" : "Benign"}</strong>
                      <span>
                        {(model.probability_malicious * 100).toFixed(1)}% malicious
                        {correct ? "" : " (differs from reference)"}
                      </span>
                    </div>
                    <ol className="cmp-features">
                      {model.top_features.map((f) => {
                        const shared = f.shared_with.length > 0;
                        return (
                          <li key={f.feature} className={shared ? "cmp-feature cmp-shared" : "cmp-feature"}>
                            <span className="cmp-rank">{f.rank}</span>
                            <div className="cmp-feature-body">
                              <span className="cmp-feature-name">{f.feature}</span>
                              <div className="feature-bar-track">
                                <div
                                  className={`feature-bar ${
                                    f.shap_value > 0 ? "feature-bar-malicious" : "feature-bar-benign"
                                  }`}
                                  style={{ width: `${(Math.abs(f.shap_value) / maxAbs) * 100}%` }}
                                />
                              </div>
                              {shared && (
                                <span className="cmp-tag">
                                  also in top 20 of {f.shared_with.map((o) => MODEL_LABELS[o]).join(" and ")}
                                </span>
                              )}
                            </div>
                          </li>
                        );
                      })}
                    </ol>
                  </div>
                );
              })}
            </div>
            <p className="note">
              Bars are scaled within each model, because the models report SHAP values on different
              scales. Red pushes toward malicious and green toward benign. Highlighted rows are
              features that at least one other model also ranks in its top 20.
            </p>
          </div>
        )}
      </section>
    </div>
  );
}

export default CompareView;
