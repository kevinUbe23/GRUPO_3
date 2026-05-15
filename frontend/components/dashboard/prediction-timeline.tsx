"use client";

import { LineChart, Target } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { compactDate, percent, priorityClass } from "@/lib/formatters";
import type { Invoice, PredictionDailyItem, PredictionHistoryItem } from "@/lib/types";
import { cn } from "@/lib/utils";

type PredictionTimelineProps = {
  invoice: Invoice | null;
  history: PredictionHistoryItem[];
  daily: PredictionDailyItem[];
  cutoffDate: string;
};

function actualLabel(invoice: Invoice | null, fallback: PredictionHistoryItem | undefined) {
  const target = invoice?.target_mora_simulado ?? fallback?.target_mora_simulado;
  const days = invoice?.dias_mora_real ?? fallback?.dias_mora_real;
  if (target) return target;
  if (days === null || days === undefined) return "Realidad pendiente";
  if (days <= 0) return "Pago dentro del plazo";
  if (days <= 30) return "Atraso leve real";
  if (days <= 60) return "Atraso alto real";
  return "Atraso critico real";
}

function points(values: number[], width: number, height: number, maxValue: number) {
  if (values.length === 0) return "";
  const step = values.length > 1 ? width / (values.length - 1) : width;
  return values
    .map((value, index) => {
      const x = values.length > 1 ? index * step : width;
      const y = height - (Math.max(0, Math.min(value, maxValue)) / maxValue) * height;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

export function PredictionTimeline({ invoice, history, daily, cutoffDate }: PredictionTimelineProps) {
  const visibleHistory = history.filter((item) => item.fecha_corte <= cutoffDate);
  const compactHistory = visibleHistory.slice(-8);
  const last = compactHistory.at(-1);
  const chartWidth = 360;
  const chartHeight = 160;
  const scorePoints = points(
    daily.map((item) => item.priority_score_0_100),
    chartWidth,
    chartHeight,
    100
  );
  const latePoints = points(
    daily.map((item) => item.any_late_probability * 100),
    chartWidth,
    chartHeight,
    100
  );
  const severePoints = points(
    daily.map((item) => item.high_risk_probability * 100),
    chartWidth,
    chartHeight,
    100
  );
  const lastDaily = daily.at(-1);

  return (
    <section className="mt-4 rounded-md border p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <LineChart className="size-5 text-muted-foreground" />
          <span className="text-sm font-semibold">Evolucion de prediccion</span>
        </div>
        <Badge variant="secondary">{daily.length || visibleHistory.length} puntos</Badge>
      </div>

      {daily.length > 0 ? (
        <div className="flex flex-col gap-3">
          <div className="overflow-hidden rounded-md border bg-background p-3">
            <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} role="img" aria-label="Evolucion diaria de prediccion" className="h-44 w-full">
              <line x1="0" y1="40" x2={chartWidth} y2="40" className="stroke-muted" strokeWidth="1" />
              <line x1="0" y1="80" x2={chartWidth} y2="80" className="stroke-muted" strokeWidth="1" />
              <line x1="0" y1="120" x2={chartWidth} y2="120" className="stroke-muted" strokeWidth="1" />
              <polyline points={latePoints} fill="none" className="stroke-muted-foreground" strokeWidth="2" />
              <polyline points={severePoints} fill="none" className="stroke-destructive" strokeWidth="2" />
              <polyline points={scorePoints} fill="none" className="stroke-primary" strokeWidth="3" />
              {daily.map((item, index) => {
                const x = daily.length > 1 ? (index * chartWidth) / (daily.length - 1) : chartWidth;
                const y = chartHeight - (Math.max(0, Math.min(item.priority_score_0_100, 100)) / 100) * chartHeight;
                return <circle key={item.fecha_corte} cx={x} cy={y} r="2.5" className="fill-primary" />;
              })}
            </svg>
            <div className="mt-2 flex flex-wrap gap-3 text-xs text-muted-foreground">
              <span>Score prioridad</span>
              <span>Prob. atraso</span>
              <span>Prob. mora grave</span>
            </div>
          </div>

          <div className="grid gap-2 sm:grid-cols-3">
            <div className="rounded-md bg-muted/45 p-3">
              <p className="text-xs uppercase text-muted-foreground">Score final</p>
              <Badge variant="outline" className={cn("mt-2 font-semibold", priorityClass(lastDaily?.priority_score_0_100 ?? 0))}>
                {(lastDaily?.priority_score_0_100 ?? 0).toFixed(1)}
              </Badge>
            </div>
            <div className="rounded-md bg-muted/45 p-3">
              <p className="text-xs uppercase text-muted-foreground">Atraso</p>
              <p className="mt-2 font-semibold">{percent.format(lastDaily?.any_late_probability ?? 0)}</p>
            </div>
            <div className="rounded-md bg-muted/45 p-3">
              <p className="text-xs uppercase text-muted-foreground">Mora grave</p>
              <p className="mt-2 font-semibold">{percent.format(lastDaily?.high_risk_probability ?? 0)}</p>
            </div>
          </div>

          <div className="rounded-md bg-muted/45 p-3">
            <div className="mb-2 flex items-center gap-2">
              <Target className="size-4 text-muted-foreground" />
              <p className="text-sm font-semibold">Comparacion contra realidad</p>
            </div>
            <p className="text-sm leading-5 text-muted-foreground">
              Ultima lectura: {lastDaily ? `${lastDaily.predicted_label_usuario} (${percent.format(lastDaily.any_late_probability)} de atraso)` : "sin lectura"}.
              Realidad: {actualLabel(invoice, last)}.
            </p>
          </div>
        </div>
      ) : compactHistory.length > 0 ? (
        <div className="grid gap-2">
          {compactHistory.map((item) => (
            <div key={item.prediccion_id} className="grid grid-cols-[88px_minmax(0,1fr)_56px] items-center gap-2">
              <span className="text-xs text-muted-foreground">{compactDate(item.fecha_corte)}</span>
              <div className="min-w-0">
                <div className="h-2 overflow-hidden rounded-full bg-muted">
                  <div
                    className="h-full rounded-full bg-primary"
                    style={{ width: `${Math.max(2, item.priority_score_0_100)}%` }}
                  />
                </div>
                <p className="mt-1 truncate text-xs text-muted-foreground">{item.predicted_label_usuario}</p>
              </div>
              <Badge variant="outline" className={cn("justify-center font-semibold", priorityClass(item.priority_score_0_100))}>
                {item.priority_score_0_100.toFixed(0)}
              </Badge>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-sm leading-5 text-muted-foreground">
          Aun no hay historial persistido hasta este corte. Recalcula la cartera en distintas fechas de corte para ver
          como se mueve el score antes de comparar contra el resultado real.
        </p>
      )}
    </section>
  );
}
