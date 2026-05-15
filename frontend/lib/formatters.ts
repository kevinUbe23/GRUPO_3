export const DEFAULT_CUTOFF = "2023-01-30";

export const money = new Intl.NumberFormat("es-EC", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0
});

export const percent = new Intl.NumberFormat("es-EC", {
  style: "percent",
  maximumFractionDigits: 0
});

export function compactDate(value: string | null) {
  if (!value) return "Sin fecha";

  return new Intl.DateTimeFormat("es-EC", {
    month: "short",
    day: "2-digit",
    year: "numeric"
  }).format(new Date(`${value}T00:00:00`));
}

export function riskClass(score: number) {
  if (score >= 80) return "border-destructive/20 bg-destructive/10 text-destructive";
  if (score >= 60) return "border-primary/20 bg-primary/10 text-primary";
  if (score >= 40) return "border-muted-foreground/20 bg-muted text-foreground";
  return "border-border bg-background text-muted-foreground";
}
