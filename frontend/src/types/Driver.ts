export interface Driver {
  driver: string;
  position: number;
  points: number;
  wins: number;
  podiums: number;
  win_rate: number;
  podium_rate: number;
  top_10_rate: number;
  experience: number;
  empirical_percentage: number;
  prediction: number;
  nationality: string;
  dob?: string;
  age?: number;
  team: string;
}