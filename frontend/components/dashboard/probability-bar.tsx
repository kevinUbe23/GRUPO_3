import { MetricHelp } from "@/components/dashboard/metric-help";
import { percent } from "@/lib/formatters";
import { cn } from "@/lib/utils";

type ProbabilityBarProps = {
  label: string;
  value: number;
  className: string;
  description?: string;
};

export function ProbabilityBar({ label, value, className, description }: ProbabilityBarProps) {
  const boundedValue = Math.min(Math.max(value, 0), 1);

  return (
    <div>
      <div className="mb-1 flex items-center justify-between gap-3 text-xs">
        <span className="inline-flex items-center gap-1 font-medium text-muted-foreground">
          {label}
          {description && <MetricHelp label={label} description={description} />}
        </span>
        <span className="tabular-nums text-muted-foreground">{percent.format(boundedValue)}</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-muted">
        <div className={cn("h-full rounded-full", className)} style={{ width: `${boundedValue * 100}%` }} />
      </div>
    </div>
  );
}
