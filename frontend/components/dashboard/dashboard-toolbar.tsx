"use client";

import { CalendarDays, Database, Loader2, RefreshCw } from "lucide-react";

import { CreateInvoiceSheet } from "@/components/dashboard/create-invoice-sheet";
import { MobileSidebar } from "@/components/dashboard/app-sidebar";
import type { DashboardView } from "@/components/dashboard/app-sidebar";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import type { InvoiceCreateInput } from "@/lib/types";

type DashboardToolbarProps = {
  fechaCorte: string;
  loading: boolean;
  onFechaCorteChange: (value: string) => void;
  onInitialize: () => void;
  onRecalculate: () => void;
  onCreateInvoice: (payload: InvoiceCreateInput) => Promise<void>;
  activeView: DashboardView;
  onViewChange: (view: DashboardView) => void;
};

export function DashboardToolbar({
  fechaCorte,
  loading,
  onFechaCorteChange,
  onInitialize,
  onRecalculate,
  onCreateInvoice,
  activeView,
  onViewChange
}: DashboardToolbarProps) {
  return (
    <header className="mb-5 flex flex-col gap-4 border-b pb-5 xl:flex-row xl:items-center xl:justify-between">
      <div className="flex min-w-0 items-start gap-3">
        <MobileSidebar
          className="mt-1 lg:hidden"
          activeView={activeView}
          onViewChange={onViewChange}
        />
        <div className="min-w-0">
          <p className="text-sm font-medium text-muted-foreground">Priorizacion inteligente</p>
          <h1 className="mt-1 text-2xl font-semibold tracking-normal text-foreground md:text-3xl">
            Cartera de cobranzas
          </h1>
        </div>
      </div>

      <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
        <label className="relative flex items-center">
          <CalendarDays className="pointer-events-none absolute left-2.5 size-4 text-muted-foreground" />
          <Input
            type="date"
            value={fechaCorte}
            onChange={(event) => onFechaCorteChange(event.target.value)}
            aria-label="Fecha de corte"
            className="h-9 min-w-40 pl-8"
          />
        </label>
        <Button onClick={onInitialize} disabled={loading} variant="outline">
          <Database data-icon="inline-start" />
          Inicializar
        </Button>
        <CreateInvoiceSheet fechaCorte={fechaCorte} disabled={loading} onCreate={onCreateInvoice} />
        <Button onClick={onRecalculate} disabled={loading}>
          {loading ? <Loader2 data-icon="inline-start" className="animate-spin" /> : <RefreshCw data-icon="inline-start" />}
          Recalcular
        </Button>
      </div>
    </header>
  );
}
