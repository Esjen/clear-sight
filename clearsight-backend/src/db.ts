/**
 * db.ts
 *
 * Sets up a single, shared PostgreSQL connection pool for the app, and
 * provides a small helper to save a prediction result after it's been
 * computed, plus a helper to fetch recent prediction history.
 */

import { Pool } from "pg";
import * as dotenv from "dotenv";

dotenv.config();

export const pool = new Pool({
  host: process.env.DB_HOST || "localhost",
  port: Number(process.env.DB_PORT) || 5432,
  database: process.env.DB_NAME || "clearsight",
  user: process.env.DB_USER || "postgres",
  password: process.env.DB_PASSWORD || "",
});

export interface PredictionRecord {
  row_index?: number;
  model: string;
  prediction: string;
  confidence: number;
  probability_malicious: number;
  true_label?: string;
  top_features: unknown;
}

export async function savePrediction(record: PredictionRecord): Promise<void> {
  const query = `
    INSERT INTO predictions
      (row_index, model, prediction, confidence, probability_malicious, true_label, top_features)
    VALUES ($1, $2, $3, $4, $5, $6, $7)
  `;
  const values = [
    record.row_index ?? null,
    record.model,
    record.prediction,
    record.confidence,
    record.probability_malicious,
    record.true_label ?? null,
    JSON.stringify(record.top_features),
  ];

  await pool.query(query, values);
}

export async function getRecentPredictions(limit = 20) {
  const result = await pool.query(
    `SELECT id, row_index, model, prediction, confidence, probability_malicious,
            true_label, created_at
     FROM predictions
     ORDER BY created_at DESC
     LIMIT $1`,
    [limit]
  );
  return result.rows;
}
