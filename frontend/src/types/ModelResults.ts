import type { Driver } from "./Driver";

export interface ModelResults {
  model_name: string;
  predictions: Driver[];
  accuracy_metrics: { total_predictions: number };
}
