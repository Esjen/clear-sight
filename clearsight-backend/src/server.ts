/**
 * server.ts
 *
 * ClearSight backend. Exposes a single main endpoint, POST /predict,
 * which:
 *   1. Receives a request body of either { row: number, model: string }
 *      (for demo purposes, using a real file from the EMBER test set)
 *      or { features: number[], model: string } (a real feature vector)
 *   2. Writes that payload to a temporary JSON file
 *   3. Calls infer.py as a subprocess, passing the temp file path
 *   4. Reads infer.py's JSON output from stdout and returns it as the
 *      API response
 *   5. Deletes the temporary file
 *
 * This "Node.js calls out to a Python subprocess" pattern is a
 * deliberate, documented design decision: the trained models are
 * Python objects (scikit-learn / XGBoost / LightGBM), so inference
 * naturally happens in Python, while the web-facing API and future
 * database storage are handled in Node/TypeScript.
 */

import express, { Request, Response } from "express";
import cors from "cors";
import { execFile } from "child_process";
import { writeFileSync, unlinkSync, readFileSync, existsSync } from "fs";
import { join } from "path";
import { randomUUID } from "crypto";
import { savePrediction, getRecentPredictions } from "./db";

const app = express();
const PORT = 3001;

// Adjust this path if your Python script lives somewhere else relative
// to where this server runs from.
const PYTHON_EXECUTABLE = "python"; // or full path to venv's python.exe if needed
const INFER_SCRIPT_PATH = join(__dirname, "..", "..", "notebooks", "infer.py");
const TEMP_DIR = join(__dirname, "..", "temp");
const NOTEBOOKS_DIR = join(__dirname, "..", "..", "notebooks");
const COMPARE_SCRIPT_PATH = join(NOTEBOOKS_DIR, "compare.py");
const AGREEMENT_PATH = join(__dirname, "..", "..", "results", "agreement_summary.json");

app.use(cors());
app.use(express.json());

app.get("/health", (req: Request, res: Response) => {
  res.json({ status: "ok" });
});

app.post("/predict", (req: Request, res: Response) => {
  const { row, features, model } = req.body;

  if (row === undefined && features === undefined) {
    return res.status(400).json({
      error: "Request body must include either 'row' (number) or 'features' (array of 2381 numbers).",
    });
  }

  const payload: Record<string, unknown> = {
    model: model || "lightgbm",
  };
  if (row !== undefined) payload.row = row;
  if (features !== undefined) payload.features = features;

  const tempFilePath = join(TEMP_DIR, `${randomUUID()}.json`);

  try {
    writeFileSync(tempFilePath, JSON.stringify(payload));
  } catch (err) {
    console.error("Failed to write temp file:", err);
    return res.status(500).json({ error: "Internal error preparing request." });
  }

  execFile(
    PYTHON_EXECUTABLE,
    [INFER_SCRIPT_PATH, "--input", tempFilePath],
    { cwd: join(__dirname, "..", "..", "notebooks"), maxBuffer: 10 * 1024 * 1024 },
    (error, stdout, stderr) => {
      // Clean up the temp file regardless of outcome
      try {
        unlinkSync(tempFilePath);
      } catch {
        // ignore cleanup errors
      }

      if (error) {
        console.error("Python inference failed:", stderr || error.message);
        return res.status(500).json({
          error: "Inference failed.",
          details: stderr || error.message,
        });
      }

      try {
        const result = JSON.parse(stdout);

        // Save to database, but don't let a DB problem break the API
        // response -- the prediction itself already succeeded.
        savePrediction({
          row_index: row,
          model: result.model,
          prediction: result.prediction,
          confidence: result.confidence,
          probability_malicious: result.probability_malicious,
          true_label: result.true_label,
          top_features: result.top_features,
        }).catch((dbErr) => {
          console.error("Failed to save prediction to database:", dbErr);
        });

        return res.json(result);
      } catch (parseErr) {
        console.error("Failed to parse Python output:", stdout);
        return res.status(500).json({
          error: "Could not parse inference result.",
          raw: stdout,
        });
      }
    }
  );
});

app.get("/history", async (req: Request, res: Response) => {
  try {
    const limit = req.query.limit ? Number(req.query.limit) : 20;
    const rows = await getRecentPredictions(limit);
    res.json(rows);
  } catch (err) {
    console.error("Failed to fetch history:", err);
    res.status(500).json({ error: "Could not fetch prediction history." });
  }
});


// ---------------------------------------------------------------------------
// Model comparison (added for the explanation-consistency feature)
// ---------------------------------------------------------------------------

// Overall agreement between the three models, with bootstrap confidence
// intervals. Produced by bootstrap_agreement.py; this endpoint only serves it.
app.get("/agreement", (req: Request, res: Response) => {
  if (!existsSync(AGREEMENT_PATH)) {
    return res.status(404).json({
      error: "agreement_not_ready",
      message: "agreement_summary.json not found. Run build_comparison_data.py and bootstrap_agreement.py first.",
    });
  }
  try {
    res.json(JSON.parse(readFileSync(AGREEMENT_PATH, "utf-8")));
  } catch (err) {
    console.error("Failed to read agreement summary:", err);
    res.status(500).json({ error: "Could not read the agreement summary." });
  }
});

// Per-file comparison of all three models. Served from precomputed SHAP
// arrays by compare.py, so it answers quickly and loads no models.
//   GET /compare?row=1234       a file from the 1,000-file explanation sample
//   GET /compare?random=1       a random file from that sample
app.get("/compare", (req: Request, res: Response) => {
  const args = [COMPARE_SCRIPT_PATH];
  if (req.query.random) {
    args.push("--random");
  } else if (req.query.row !== undefined && /^\d+$/.test(String(req.query.row))) {
    args.push("--row", String(req.query.row));
  } else {
    return res.status(400).json({ error: "bad_request", message: "Provide ?row=<number> or ?random=1." });
  }

  execFile(
    PYTHON_EXECUTABLE,
    args,
    { cwd: NOTEBOOKS_DIR, timeout: 30000, maxBuffer: 5 * 1024 * 1024 },
    (error, stdout, stderr) => {
      if (error) {
        console.error("compare.py failed:", stderr || error.message);
        return res.status(500).json({ error: "compare_failed", message: stderr || error.message });
      }
      try {
        const result = JSON.parse(stdout);
        if (result.error === "not_in_sample") return res.status(404).json(result);
        if (result.error) return res.status(500).json(result);
        return res.json(result);
      } catch {
        console.error("Could not parse compare.py output:", stdout);
        return res.status(500).json({ error: "compare_failed", message: "Could not parse comparison result." });
      }
    }
  );
});

app.listen(PORT, () => {
  console.log(`ClearSight backend running at http://localhost:${PORT}`);
});
