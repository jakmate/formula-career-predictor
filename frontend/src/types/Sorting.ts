export type SortField = 'driver' | 'position' | 'points' | 'win_rate' | 'podium_rate' | 'top_10_rate' | 'dnf_rate' |
                         'participation_rate' | 'experience' | 'empirical_percentage';
export type SortDirection = 'asc' | 'desc';

export interface SortConfig {
  field: SortField;
  direction: SortDirection;
}
