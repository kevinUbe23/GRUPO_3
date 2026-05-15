"use client";

import { FileText, Gauge, Loader2, PhoneCall, UserRound } from "lucide-react";

import { ProbabilityBar } from "@/components/dashboard/probability-bar";
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
import { compactDate, money, riskClass } from "@/lib/formatters";
import type { Customer, Interaction, Invoice, Prediction, PrioritizedInvoice, Segment } from "@/lib/types";
import { cn } from "@/lib/utils";

type InvoiceDetailPanelProps = {
  selected: PrioritizedInvoice | null;
  invoice: Invoice | null;
  customer: Customer | null;
  segment: Segment | null;
  interactions: Interaction[];
  prediction: Prediction | null;
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
  detailLoading,
  onScore
}: InvoiceDetailPanelProps) {
  const score = prediction?.priority_score_0_100 ?? selected?.priority_score_0_100 ?? 0;

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
            Scoring
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
              <span className="text-sm font-semibold">Riesgo actual</span>
              <Badge variant="outline" className={cn("font-semibold", riskClass(score))}>
                {score.toFixed(1)}
              </Badge>
            </div>
            <div className="flex flex-col gap-3">
              <ProbabilityBar
                label="Pago dentro del plazo"
                value={prediction?.prob_pago_plazo ?? Math.max(0, 1 - selected.any_late_probability)}
                className="bg-primary"
              />
              <ProbabilityBar
                label="Cualquier atraso"
                value={prediction?.any_late_probability ?? selected.any_late_probability}
                className="bg-muted-foreground"
              />
              <ProbabilityBar
                label="Atraso alto o critico"
                value={prediction?.high_risk_probability ?? selected.high_risk_probability}
                className="bg-destructive"
              />
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
              <p className="mt-1 text-sm text-muted-foreground">Riesgo cliente {segment.riesgo_0_100.toFixed(1)}/100</p>
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
