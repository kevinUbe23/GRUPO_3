"use client";

import { FileText, Gauge, Loader2, PhoneCall, UserRound } from "lucide-react";

import { MetricHelp } from "@/components/dashboard/metric-help";
import { ProbabilityBar } from "@/components/dashboard/probability-bar";
import { PredictionTimeline } from "@/components/dashboard/prediction-timeline";
import { Stars } from "@/components/dashboard/stars";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle
} from "@/components/ui/empty";
import { clientRiskLevel, compactDate, money, priorityActionHint, priorityClass, priorityLevel } from "@/lib/formatters";
import type { Customer, Interaction, Invoice, Prediction, PredictionHistoryItem, PrioritizedInvoice, Segment } from "@/lib/types";
import { cn } from "@/lib/utils";

type InvoiceDetailPanelProps = {
  selected: PrioritizedInvoice | null;
  invoice: Invoice | null;
  customer: Customer | null;
  segment: Segment | null;
  interactions: Interaction[];
  prediction: Prediction | null;
  predictionHistory: PredictionHistoryItem[];
  detailLoading: boolean;
  onScore: () => void;
};

export function InvoiceDetailPanel({
  selected,
  invoice,
  customer,
  segment,
  interactions,
  prediction,
  predictionHistory,
  detailLoading,
  onScore
}: InvoiceDetailPanelProps) {
  const score = prediction?.priority_score_0_100 ?? selected?.priority_score_0_100 ?? 0;
  const predictedLabel = prediction?.predicted_label_usuario ?? selected?.predicted_label_usuario;
  const probabilityValues = {
    pagoPlazo: prediction?.prob_pago_plazo ?? selected?.prob_pago_plazo ?? Math.max(0, 1 - (selected?.any_late_probability ?? 0)),
    atrasoLeve: prediction?.prob_atraso_leve ?? selected?.prob_atraso_leve ?? 0,
    atrasoAlto: prediction?.prob_atraso_alto ?? selected?.prob_atraso_alto ?? 0,
    atrasoCritico: prediction?.prob_atraso_critico ?? selected?.prob_atraso_critico ?? 0,
    anyLate: prediction?.any_late_probability ?? selected?.any_late_probability ?? 0,
    highRisk: prediction?.high_risk_probability ?? selected?.high_risk_probability ?? 0
  };
  const scoreDescription =
    "Indice operativo de 0 a 100 para ordenar la cola: 0-39 baja, 40-59 media, 60-79 alta y 80+ critica. No es una probabilidad; indica que tan pronto conviene gestionar la factura.";
  const clientRiskDescription =
    "Metrica historica del cliente, distinta al score de prioridad de la factura. Resume comportamiento pasado: mora, cumplimiento, promesas y contacto. Sirve como contexto, no como orden directo de la cola.";

  return (
    <Card id="detalle" className="xl:sticky xl:top-5 xl:self-start">
      <CardHeader className="border-b p-4">
        <div className="flex items-center justify-between gap-3">
          <div className="min-w-0">
            <CardTitle className="truncate text-base">{selected?.factura_id ?? "Detalle de factura"}</CardTitle>
            <p className="mt-1 truncate text-sm text-muted-foreground">
              {customer ? `${customer.nombre} / ${customer.sector}` : "Selecciona un caso"}
            </p>
          </div>
          <Button onClick={onScore} disabled={!selected || detailLoading} className="shrink-0">
            {detailLoading ? <Loader2 data-icon="inline-start" className="animate-spin" /> : <Gauge data-icon="inline-start" />}
            Actualizar prediccion
          </Button>
        </div>
      </CardHeader>

      {selected ? (
        <CardContent className="max-h-[740px] overflow-auto p-4 scrollbar-thin">
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-md border bg-muted/40 p-3">
              <p className="text-xs uppercase text-muted-foreground">Monto</p>
              <p className="mt-1 text-lg font-semibold">{money.format(invoice?.monto ?? selected.monto)}</p>
            </div>
            <div className="rounded-md border bg-muted/40 p-3">
              <p className="text-xs uppercase text-muted-foreground">Vencimiento</p>
              <p className="mt-1 text-lg font-semibold">
                {compactDate(invoice?.fecha_vencimiento ?? selected.fecha_vencimiento)}
              </p>
            </div>
          </div>

          <section className="mt-4 rounded-md border p-4">
            <div className="mb-3 flex items-center justify-between">
              <span className="inline-flex items-center gap-1 text-sm font-semibold">
                Score de prioridad
                <MetricHelp label="Score de prioridad" description={scoreDescription} />
              </span>
              <Badge variant="outline" className={cn("font-semibold", priorityClass(score))}>
                {score.toFixed(1)}/100
              </Badge>
            </div>
            <p className="mb-3 text-sm text-muted-foreground">
              Prioridad {priorityLevel(score).toLowerCase()}: {priorityActionHint(score).toLowerCase()}.
            </p>
            {predictedLabel && (
              <div className="mb-3 rounded-md bg-muted/45 px-3 py-2 text-sm">
                <span className="font-medium">Clase predicha: </span>
                <span className="text-muted-foreground">{predictedLabel}</span>
              </div>
            )}
            <div className="flex flex-col gap-3">
              <ProbabilityBar
                label="Pago dentro del plazo"
                value={probabilityValues.pagoPlazo}
                className="bg-primary"
                description="Probabilidad de que la factura se pague sin atraso relevante frente al vencimiento pactado."
              />
              <ProbabilityBar
                label="Atraso leve"
                value={probabilityValues.atrasoLeve}
                className="bg-muted-foreground"
                description="Probabilidad de pago hasta 30 dias despues del vencimiento."
              />
              <ProbabilityBar
                label="Atraso alto"
                value={probabilityValues.atrasoAlto}
                className="bg-muted-foreground"
                description="Probabilidad de pago entre 31 y 60 dias despues del vencimiento."
              />
              <ProbabilityBar
                label="Atraso critico / 60+"
                value={probabilityValues.atrasoCritico}
                className="bg-destructive"
                description="Probabilidad de mora mayor a 60 dias. La clase tecnica del modelo es +90, pero en negocio se interpreta como 60+ dias."
              />
            </div>
            <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-muted-foreground">
              <div className="rounded-md bg-muted/35 px-3 py-2">
                Cualquier atraso: {Math.round(probabilityValues.anyLate * 100)}%
              </div>
              <div className="rounded-md bg-muted/35 px-3 py-2">
                Mora grave: {Math.round(probabilityValues.highRisk * 100)}%
              </div>
            </div>
          </section>

          <section className="mt-4 rounded-md border p-4">
            <div className="flex items-start gap-3">
              <PhoneCall className="mt-0.5 size-5 text-muted-foreground" />
              <div>
                <p className="text-sm font-semibold">
                  {prediction?.accion_sugerida.nombre ?? selected.accion_sugerida}
                </p>
                <p className="mt-1 text-sm leading-5 text-muted-foreground">
                  {prediction?.accion_sugerida.motivo ?? selected.predicted_label_usuario}
                </p>
              </div>
            </div>
          </section>

          <PredictionTimeline invoice={invoice} history={predictionHistory} cutoffDate={selected.fecha_corte} />

          {segment && (
            <section className="mt-4 rounded-md border p-4">
              <div className="mb-3 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <UserRound className="size-5 text-muted-foreground" />
                  <span className="text-sm font-semibold">Perfil del cliente</span>
                </div>
                <Stars value={segment.rating_estrellas} />
              </div>
              <p className="text-sm font-medium">{segment.tipo_cliente}</p>
              <p className="mt-1 inline-flex items-center gap-1 text-sm text-muted-foreground">
                Riesgo historico del cliente {segment.riesgo_0_100.toFixed(1)}/100
                <MetricHelp label="Riesgo historico del cliente" description={clientRiskDescription} />
              </p>
              <p className="mt-1 text-xs text-muted-foreground">Nivel historico: {clientRiskLevel(segment.riesgo_0_100)}</p>
              <p className="mt-3 text-sm leading-5 text-muted-foreground">{segment.por_que_rating}</p>
            </section>
          )}

          <section className="mt-4 rounded-md border p-4">
            <p className="mb-3 text-sm font-semibold">Gestiones</p>
            <div className="flex flex-col gap-3">
              {interactions.slice(-5).reverse().map((item) => (
                <div key={item.gestion_id} className="border-l-2 border-border pl-3">
                  <div className="flex items-center justify-between gap-2 text-sm">
                    <span className="font-medium">{item.resultado}</span>
                    <span className="text-xs text-muted-foreground">{compactDate(item.fecha_gestion)}</span>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {item.canal} / {item.contacto_exitoso ? "contacto exitoso" : "sin contacto"}
                  </p>
                </div>
              ))}
              {interactions.length === 0 && <p className="text-sm text-muted-foreground">No hay gestiones registradas.</p>}
            </div>
          </section>
        </CardContent>
      ) : (
        <CardContent className="flex min-h-[460px] items-center justify-center p-8">
          <Empty className="border-0">
            <EmptyHeader>
              <EmptyMedia variant="icon">
                <FileText />
              </EmptyMedia>
              <EmptyTitle>Detalle de factura</EmptyTitle>
              <EmptyDescription>Selecciona una factura de la cola.</EmptyDescription>
            </EmptyHeader>
          </Empty>
        </CardContent>
      )}
    </Card>
  );
}
