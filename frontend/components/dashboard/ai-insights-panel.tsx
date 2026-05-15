"use client";

import { Activity, AlertTriangle, BarChart3, GitCompareArrows } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { percent } from "@/lib/formatters";
import type { PrioritizedInvoice } from "@/lib/types";

type AiInsightsPanelProps = {
  rows: PrioritizedInvoice[];
};

function average(values: number[]) {
  if (values.length === 0) return 0;
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function priorityBucket(score: number) {
  if (score >= 80) return "Critica";
  if (score >= 60) return "Alta";
  if (score >= 40) return "Media";
  return "Baja";
}

export function AiInsightsPanel({ rows }: AiInsightsPanelProps) {
  const total = rows.length;
  const scoredRows = rows.filter((row) => row.priority_score_0_100 !== null && row.priority_score_0_100 !== undefined);
  const riskRows = rows.filter((row) => row.any_late_probability !== null && row.any_late_probability !== undefined);
  const highRiskRows = rows.filter((row) => row.high_risk_probability !== null && row.high_risk_probability !== undefined);
  const critical = scoredRows.filter((row) => (row.priority_score_0_100 ?? 0) >= 80).length;
  const highOrCritical = scoredRows.filter((row) => (row.priority_score_0_100 ?? 0) >= 60).length;
  const averageLateRisk = average(riskRows.map((row) => row.any_late_probability ?? 0));
  const averageHighRisk = average(highRiskRows.map((row) => row.high_risk_probability ?? 0));
  const topShare = rows
    .slice(0, 10)
    .reduce((totalAmount, row) => totalAmount + row.monto, 0) / Math.max(rows.reduce((totalAmount, row) => totalAmount + row.monto, 0), 1);

  const buckets = ["Critica", "Alta", "Media", "Baja"].map((label) => {
    const count = scoredRows.filter((row) => priorityBucket(row.priority_score_0_100 ?? 0) === label).length;
    return { label, count, share: scoredRows.length ? count / scoredRows.length : 0 };
  });

  const labels = Array.from(new Set(rows.map((row) => row.predicted_label_usuario).filter(Boolean))).slice(0, 4);

  return (
    <section id="auditoria-ia" className="mt-4 grid gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
      <Card>
        <CardHeader className="border-b p-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <Activity className="size-5 text-muted-foreground" />
              <CardTitle className="text-base">Lectura auditora de IA</CardTitle>
            </div>
            <Badge variant="secondary">
              {scoredRows.length.toLocaleString("es-EC")} con prediccion / {total.toLocaleString("es-EC")} total
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="grid gap-4 p-4 md:grid-cols-3">
          <div className="rounded-md border bg-muted/35 p-3">
            <p className="text-xs font-medium uppercase text-muted-foreground">Atraso promedio</p>
            <p className="mt-2 text-2xl font-semibold">{percent.format(averageLateRisk)}</p>
            <p className="mt-1 text-xs text-muted-foreground">Probabilidad media de cualquier atraso.</p>
          </div>
          <div className="rounded-md border bg-muted/35 p-3">
            <p className="text-xs font-medium uppercase text-muted-foreground">Mora grave media</p>
            <p className="mt-2 text-2xl font-semibold">{percent.format(averageHighRisk)}</p>
            <p className="mt-1 text-xs text-muted-foreground">Mora alta o critica estimada.</p>
          </div>
          <div className="rounded-md border bg-muted/35 p-3">
            <p className="text-xs font-medium uppercase text-muted-foreground">Concentracion top 10</p>
            <p className="mt-2 text-2xl font-semibold">{percent.format(topShare)}</p>
            <p className="mt-1 text-xs text-muted-foreground">Peso monetario de los primeros casos.</p>
          </div>

          <div className="md:col-span-2">
            <div className="mb-3 flex items-center gap-2">
              <BarChart3 className="size-4 text-muted-foreground" />
              <p className="text-sm font-semibold">Distribucion del top cargado por score</p>
            </div>
            <div className="flex flex-col gap-3">
              {buckets.map((bucket) => (
                <div key={bucket.label}>
                  <div className="mb-1 flex items-center justify-between gap-3 text-xs">
                    <span className="font-medium text-muted-foreground">{bucket.label}</span>
                    <span className="tabular-nums text-muted-foreground">{bucket.count}</span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-muted">
                    <div className="h-full rounded-full bg-primary" style={{ width: `${bucket.share * 100}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div>
            <div className="mb-3 flex items-center gap-2">
              <GitCompareArrows className="size-4 text-muted-foreground" />
              <p className="text-sm font-semibold">Etiquetas dominantes</p>
            </div>
            <div className="flex flex-col gap-2">
              {labels.map((label) => (
                <div key={label} className="flex items-center justify-between gap-3 rounded-md border px-3 py-2 text-sm">
                  <span className="truncate">{label}</span>
                  <Badge variant="outline">{rows.filter((row) => row.predicted_label_usuario === label).length}</Badge>
                </div>
              ))}
              {labels.length === 0 && <p className="text-sm text-muted-foreground">Sin predicciones para analizar.</p>}
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="border-b p-4">
          <div className="flex items-center gap-2">
            <AlertTriangle className="size-5 text-muted-foreground" />
            <CardTitle className="text-base">Senales para revision</CardTitle>
          </div>
        </CardHeader>
        <CardContent className="flex flex-col gap-3 p-4">
          <div className="flex items-center justify-between gap-3 rounded-md border px-3 py-2">
            <span className="text-sm text-muted-foreground">Prioridad critica</span>
            <Badge variant={critical > 0 ? "destructive" : "secondary"}>{critical}</Badge>
          </div>
          <div className="flex items-center justify-between gap-3 rounded-md border px-3 py-2">
            <span className="text-sm text-muted-foreground">Prioridad alta o critica</span>
            <Badge variant="outline">{highOrCritical}</Badge>
          </div>
          <p className="text-sm leading-5 text-muted-foreground">
            Esta vista resume el top cargado de la cola activa: como concentra la prioridad, como distribuye el score y
            que etiquetas predominan.
          </p>
        </CardContent>
      </Card>
    </section>
  );
}
