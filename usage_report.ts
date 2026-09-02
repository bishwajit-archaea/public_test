export interface PlanUsage { plan: string; seatsUsed: number; }

export function overageSeats(usage: PlanUsage, included: number): number {
  return Math.max(0, usage.seatsUsed - included);
}
