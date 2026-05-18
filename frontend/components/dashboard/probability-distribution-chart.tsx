"use client";

import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts";

import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart";

type ProbabilityDistributionChartProps = {
  pagoPlazo: number | null;
  atrasoLeve: number | null;
  atrasoAlto: number | null;
  atrasoCritico: number | null;
};

const chartConfig = {
  probabilidad: { label: "Probabilidad", color: "var(--primary)" }
} satisfies ChartConfig;

export function ProbabilityDistributionChart({
  pagoPlazo,
  atrasoLeve,
  atrasoAlto,
  atrasoCritico
}: ProbabilityDistributionChartProps) {
  const rows = [
    { bucket: "Pago dentro del plazo", probabilidad: Math.round((pagoPlazo ?? 0) * 100) },
    { bucket: "Atraso leve", probabilidad: Math.round((atrasoLeve ?? 0) * 100) },
    { bucket: "Atraso alto", probabilidad: Math.round((atrasoAlto ?? 0) * 100) },
    { bucket: "Atraso critico", probabilidad: Math.round((atrasoCritico ?? 0) * 100) }
  ];

  return (
    <ChartContainer
      config={chartConfig}
      className="h-44 w-full"
      role="img"
      aria-label="Distribucion de probabilidades de pago y atraso de la factura"
    >
      <BarChart data={rows} layout="vertical" margin={{ left: 8, right: 18, top: 4, bottom: 4 }}>
        <CartesianGrid horizontal={false} />
        <XAxis type="number" domain={[0, 100]} tickLine={false} axisLine={false} unit="%" />
        <YAxis dataKey="bucket" type="category" tickLine={false} axisLine={false} width={118} />
        <ChartTooltip
          cursor={false}
          content={
            <ChartTooltipContent
              hideLabel
              formatter={(value) => (
                <div className="flex w-full items-center justify-between gap-6">
                  <span className="text-muted-foreground">Probabilidad</span>
                  <span className="font-mono font-medium tabular-nums">{Number(value).toFixed(0)}%</span>
                </div>
              )}
            />
          }
        />
        <Bar dataKey="probabilidad" fill="var(--color-probabilidad)" radius={4} />
      </BarChart>
    </ChartContainer>
  );
}
