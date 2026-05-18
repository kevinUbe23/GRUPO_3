import { CircleDollarSign, Clock3, FileText, ShieldCheck } from "lucide-react";

import { MetricCard } from "@/components/dashboard/metric-card";
import type { DashboardSummary } from "@/lib/types";
import { money } from "@/lib/formatters";

type SummaryCardsProps = {
  dashboard: DashboardSummary | null;
};

export function SummaryCards({ dashboard }: SummaryCardsProps) {
  return (
    <section id="resumen" className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
      <MetricCard
        label="Facturas activas"
        value={(dashboard?.facturas_activas ?? 0).toLocaleString("es-EC")}
        icon={FileText}
        tone="border-primary/20 bg-primary/10 text-primary"
      />
      <MetricCard
        label="Monto pendiente"
        value={money.format(dashboard?.monto_pendiente ?? 0)}
        icon={CircleDollarSign}
        tone="border-border bg-background text-foreground"
      />
      <MetricCard
        label="Monto vencido"
        value={money.format(dashboard?.monto_vencido ?? 0)}
        description={`${(dashboard?.clientes_con_monto_vencido ?? 0).toLocaleString("es-EC")} clientes con deuda vencida`}
        icon={Clock3}
        tone="border-muted-foreground/20 bg-muted text-foreground"
      />
      <MetricCard
        label="Promesas activas"
        value={(dashboard?.promesas_activas ?? 0).toLocaleString("es-EC")}
        icon={ShieldCheck}
        tone="border-border bg-muted text-muted-foreground"
      />
    </section>
  );
}
