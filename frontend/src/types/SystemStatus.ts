export interface SystemStatus {
  last_scrape?: string;
  last_training?: string;
  models_available?: string[];
  data_health?: Record<string, Record<string, number>>;
}
