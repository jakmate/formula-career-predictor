export interface SystemStatus {
  last_scrape_full?: string;
  last_scrape_predictions?: string;
  last_scrape_schedule?: string;
  last_training?: string;
  models_available?: {
    f3_to_f2?: string[];
    f2_to_f1?: string[];
  };
  data_health?: Record<string, Record<string, number>>;
}
