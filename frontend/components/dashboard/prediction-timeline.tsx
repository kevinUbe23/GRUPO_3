"use client";

import { LineChart, Target } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { compactDate, percent, priorityClass } from "@/lib/formatters";
import type { Invoice, PredictionHistoryItem } from "@/lib/types";
import { cn } from "@/lib/utils";

type PredictionTimelineProps = {
  invoice: Invoice | null;
  history: PredictionHistoryItem[];
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

export function PredictionTimeline({ invoice, history, cutoffDate }: PredictionTimelineProps) {
  const visibleHistory = history.filter((item) => item.fecha_corte <= cutoffDate);
  const compactHistory = visibleHistory.slice(-8);
  const last = compactHistory.at(-1);

  return (
    <section className="mt-4 rounded-md border p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <LineChart className="size-5 text-muted-foreground" />
          <span className="text-sm font-semibold">Evolucion de prediccion</span>
        </div>
        <Badge variant="secondary">{visibleHistory.length} cortes</Badge>
      </div>

      {compactHistory.length > 0 ? (
        <div className="flex flex-col gap-3">
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

          <div className="rounded-md bg-muted/45 p-3">
            <div className="mb-2 flex items-center gap-2">
              <Target className="size-4 text-muted-foreground" />
              <p className="text-sm font-semibold">Comparacion contra realidad</p>
            </div>
            <p className="text-sm leading-5 text-muted-foreground">
              Ultima lectura: {last ? `${last.predicted_label_usuario} (${percent.format(last.any_late_probability)} de atraso)` : "sin lectura"}.
              Realidad: {actualLabel(invoice, last)}.
            </p>
          </div>
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
