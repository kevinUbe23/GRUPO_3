import { percent } from "@/lib/formatters";
import { cn } from "@/lib/utils";

type ProbabilityBarProps = {
  label: string;
  value: number;
  className: string;
};

export function ProbabilityBar({ label, value, className }: ProbabilityBarProps) {
  return (
    <div>
      <div className="mb-1 flex items-center justify-between gap-3 text-xs">
        <span className="font-medium text-muted-foreground">{label}</span>
        <span className="tabular-nums text-muted-foreground">{percent.format(value)}</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-muted">
        <div className={cn("h-full rounded-full", className)} style={{ width: `${value * 100}%` }} />
      </div>
    </div>
  );
}
