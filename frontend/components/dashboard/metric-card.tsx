import type { LucideIcon } from "lucide-react";

import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

type MetricCardProps = {
  label: string;
  value: string;
  description?: string;
  icon: LucideIcon;
  tone: string;
};

export function MetricCard({ label, value, description, icon: Icon, tone }: MetricCardProps) {
  return (
    <Card>
      <CardContent className="flex items-start justify-between gap-3 p-4">
        <div className="min-w-0">
          <p className="text-xs font-medium uppercase text-muted-foreground">{label}</p>
          <p className="mt-2 truncate text-2xl font-semibold text-foreground">{value}</p>
          {description && <p className="mt-1 text-xs text-muted-foreground">{description}</p>}
        </div>
        <span className={cn("rounded-md border p-2", tone)}>
          <Icon className="size-5" />
        </span>
      </CardContent>
    </Card>
  );
}
