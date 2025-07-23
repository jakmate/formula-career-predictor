export interface Driver {
  driver: string;
  position: number;
  points: number;
  wins: number;
  podiums: number;
  win_rate: number;
  top_10_rate: number;
  experience: number;
  empirical_percentage: number;
  nationality: string;
  dob?: string;
  age?: number;
  avg_quali_pos?: number;
  dnf_rate: number;
  participation_rate: number;
  teammate_h2h: number;
  team: string;
  team_pos: number;
  team_points: number;
}
