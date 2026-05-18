"use client";

import { LineChart, Target } from "lucide-react";
import { CartesianGrid, Legend, Line, LineChart as RechartsLineChart, XAxis, YAxis } from "recharts";

import {
  ChartContainer,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig
} from "@/components/ui/chart";
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

const chartConfig = {
  score: { label: "Score de prioridad", color: "var(--primary)" },
  atraso: { label: "Prob. cualquier atraso", color: "var(--muted-foreground)" },
  moraGrave: { label: "Prob. mora grave", color: "var(--destructive)" }
} satisfies ChartConfig;

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

function interpretDifference(predicted: string | undefined, actual: string) {
  if (!predicted) return "Sin prediccion final para contrastar.";
  if (actual === "Realidad pendiente") return "El resultado real todavia no esta cerrado.";
  return `${predicted} frente a ${actual}.`;
}

export function PredictionTimeline({ invoice, history, daily, cutoffDate }: PredictionTimelineProps) {
  const visibleHistory = history.filter((item) => item.fecha_corte <= cutoffDate);
  const compactHistory = visibleHistory.slice(-8);
  const last = compactHistory.at(-1);
  const chartData = daily.map((item) => ({
    fecha: compactDate(item.fecha_corte),
    fechaCompleta: item.fecha_corte,
    score: Number(item.priority_score_0_100.toFixed(1)),
    atraso: Number((item.any_late_probability * 100).toFixed(1)),
    moraGrave: Number((item.high_risk_probability * 100).toFixed(1)),
    etiqueta: item.predicted_label_usuario
  }));
  const lastDaily = daily.at(-1);
  const actual = actualLabel(invoice, last);

  return (
    <section className="mt-4 rounded-md border p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <LineChart className="size-5 text-muted-foreground" />
          <span className="text-sm font-semibold">Evolucion de prediccion</span>
        </div>
        <Badge variant="secondary">{daily.length || visibleHistory.length} puntos</Badge>
      </div>

      {chartData.length > 1 ? (
        <div className="flex flex-col gap-3">
          <ChartContainer
            config={chartConfig}
            className="h-56 w-full"
            role="img"
            aria-label="Evolucion del score de prioridad y probabilidades de atraso de la factura"
          >
            <RechartsLineChart data={chartData} margin={{ left: 0, right: 12, top: 8, bottom: 0 }}>
              <CartesianGrid vertical={false} />
              <XAxis dataKey="fecha" tickLine={false} axisLine={false} minTickGap={18} />
              <YAxis tickLine={false} axisLine={false} domain={[0, 100]} width={32} />
              <ChartTooltip
                content={
                  <ChartTooltipContent
                    labelFormatter={(_, payload) => payload?.[0]?.payload?.fechaCompleta ?? ""}
                    formatter={(value, name) => {
                      const label = chartConfig[String(name) as keyof typeof chartConfig]?.label ?? name;
                      const suffix = name === "score" ? "/100" : "%";
                      return (
                        <div className="flex w-full items-center justify-between gap-6">
                          <span className="text-muted-foreground">{label}</span>
                          <span className="font-mono font-medium tabular-nums">
                            {Number(value).toFixed(1)}
                            {suffix}
                          </span>
                        </div>
                      );
                    }}
                  />
                }
              />
              <Legend content={<ChartLegendContent />} />
              <Line dataKey="score" type="monotone" stroke="var(--color-score)" strokeWidth={3} dot={false} />
              <Line dataKey="atraso" type="monotone" stroke="var(--color-atraso)" strokeWidth={2} dot={false} />
              <Line dataKey="moraGrave" type="monotone" stroke="var(--color-moraGrave)" strokeWidth={2} dot={false} />
            </RechartsLineChart>
          </ChartContainer>

          <div className="grid gap-2 sm:grid-cols-3">
            <div className="rounded-md bg-muted/45 p-3">
              <p className="text-xs uppercase text-muted-foreground">Prediccion final</p>
              <Badge variant="outline" className={cn("mt-2 font-semibold", priorityClass(lastDaily?.priority_score_0_100 ?? 0))}>
                {(lastDaily?.priority_score_0_100 ?? 0).toFixed(1)}/100
              </Badge>
            </div>
            <div className="rounded-md bg-muted/45 p-3">
              <p className="text-xs uppercase text-muted-foreground">Resultado real</p>
              <p className="mt-2 text-sm font-semibold">{actual}</p>
            </div>
            <div className="rounded-md bg-muted/45 p-3">
              <p className="text-xs uppercase text-muted-foreground">Lectura</p>
              <p className="mt-2 text-sm font-semibold">
                {percent.format(lastDaily?.any_late_probability ?? 0)} atraso
              </p>
            </div>
          </div>

          <div className="rounded-md bg-muted/45 p-3">
            <div className="mb-2 flex items-center gap-2">
              <Target className="size-4 text-muted-foreground" />
              <p className="text-sm font-semibold">Comparacion prediccion vs realidad</p>
            </div>
            <p className="text-sm leading-5 text-muted-foreground">
              {interpretDifference(lastDaily?.predicted_label_usuario, actual)}
            </p>
          </div>
        </div>
      ) : chartData.length === 1 ? (
        <div className="rounded-md bg-muted/45 p-3 text-sm text-muted-foreground">
          Prediccion final: <span className="font-medium text-foreground">{chartData[0].etiqueta}</span>, score{" "}
          <span className="font-medium text-foreground">{chartData[0].score.toFixed(1)}/100</span>. Resultado real:{" "}
          <span className="font-medium text-foreground">{actual}</span>.
        </div>
      ) : compactHistory.length > 0 ? (
        <div className="grid gap-2">
          {compactHistory.map((item) => (
            <div key={item.prediccion_id} className="grid grid-cols-[88px_minmax(0,1fr)_56px] items-center gap-2">
              <span className="text-xs text-muted-foreground">{compactDate(item.fecha_corte)}</span>
              <div className="min-w-0">
                <div className="h-2 overflow-hidden rounded-full bg-muted">
                  <div className="h-full rounded-full bg-primary" style={{ width: `${Math.max(2, item.priority_score_0_100)}%` }} />
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
