"use client";

import { Search, UsersRound } from "lucide-react";

import { Stars } from "@/components/dashboard/stars";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle
} from "@/components/ui/empty";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { clientRiskLevel } from "@/lib/formatters";
import type { Customer, Segment } from "@/lib/types";
import { cn } from "@/lib/utils";

type CustomersViewProps = {
  customers: Customer[];
  selectedCustomer: Customer | null;
  selectedSegment: Segment | null;
  query: string;
  loading: boolean;
  onQueryChange: (value: string) => void;
  onSelect: (customer: Customer) => void;
};

export function CustomersView({
  customers,
  selectedCustomer,
  selectedSegment,
  query,
  loading,
  onQueryChange,
  onSelect
}: CustomersViewProps) {
  return (
    <section className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_420px]">
      <Card className="min-w-0">
        <CardHeader className="gap-3 border-b p-4 md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-2">
            <UsersRound className="size-5 text-muted-foreground" />
            <CardTitle className="text-base">Clientes</CardTitle>
            <Badge variant="secondary">{customers.length.toLocaleString("es-EC")}</Badge>
          </div>
          <label className="relative block md:w-80">
            <Search className="pointer-events-none absolute left-2.5 top-2 size-4 text-muted-foreground" />
            <Input
              value={query}
              onChange={(event) => onQueryChange(event.target.value)}
              placeholder="Buscar cliente o sector"
              aria-label="Buscar cliente o sector"
              className="pl-8"
            />
          </label>
        </CardHeader>
        <CardContent className="p-0">
          <div className="max-h-[680px] overflow-auto scrollbar-thin">
            {customers.map((customer) => (
              <button
                key={customer.cliente_id}
                type="button"
                className={cn(
                  "grid w-full grid-cols-[minmax(0,1fr)_120px] gap-3 border-b px-4 py-3 text-left transition-colors hover:bg-muted/45",
                  selectedCustomer?.cliente_id === customer.cliente_id && "bg-accent"
                )}
                onClick={() => onSelect(customer)}
              >
                <span className="min-w-0">
                  <span className="block truncate font-medium">{customer.nombre}</span>
                  <span className="mt-1 block text-xs text-muted-foreground">
                    {customer.cliente_id} / {customer.sector}
                  </span>
                </span>
                <span className="text-right text-sm text-muted-foreground">
                  {customer.antiguedad_meses.toLocaleString("es-EC")} meses
                </span>
              </button>
            ))}
            {customers.length === 0 && (
              <Empty className="min-h-[300px] border-0">
                <EmptyHeader>
                  <EmptyMedia variant="icon">
                    <UsersRound />
                  </EmptyMedia>
                  <EmptyTitle>{loading ? "Cargando clientes" : "Sin clientes"}</EmptyTitle>
                  <EmptyDescription>No hay clientes que coincidan con la busqueda.</EmptyDescription>
                </EmptyHeader>
              </Empty>
            )}
          </div>
        </CardContent>
      </Card>

      <Card className="xl:sticky xl:top-5 xl:self-start">
        <CardHeader className="border-b p-4">
          <CardTitle className="text-base">{selectedCustomer?.nombre ?? "Detalle de cliente"}</CardTitle>
          <p className="text-sm text-muted-foreground">
            {selectedCustomer ? `${selectedCustomer.cliente_id} / ${selectedCustomer.sector}` : "Selecciona un cliente"}
          </p>
        </CardHeader>
        <CardContent className="p-4">
          {selectedCustomer && selectedSegment ? (
            <div className="flex flex-col gap-4">
              <div className="grid grid-cols-2 gap-3">
                <div className="rounded-md border bg-muted/35 p-3">
                  <p className="text-xs uppercase text-muted-foreground">Riesgo historico</p>
                  <p className="mt-1 text-2xl font-semibold">{selectedSegment.riesgo_0_100.toFixed(1)}</p>
                  <p className="text-xs text-muted-foreground">{clientRiskLevel(selectedSegment.riesgo_0_100)}</p>
                </div>
                <div className="rounded-md border bg-muted/35 p-3">
                  <p className="text-xs uppercase text-muted-foreground">Rating</p>
                  <div className="mt-2">
                    <Stars value={selectedSegment.rating_estrellas} />
                  </div>
                  <p className="mt-1 text-xs text-muted-foreground">{selectedSegment.rating_label}</p>
                </div>
              </div>
              <div>
                <p className="text-sm font-semibold">{selectedSegment.tipo_cliente}</p>
                <p className="mt-2 text-sm leading-5 text-muted-foreground">{selectedSegment.por_que_rating}</p>
              </div>
              <Separator />
              <div>
                <p className="text-sm font-semibold">Segmento operativo</p>
                <p className="mt-2 text-sm leading-5 text-muted-foreground">{selectedSegment.por_que_cluster}</p>
              </div>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="rounded-md border p-3">
                  <p className="text-xs uppercase text-muted-foreground">Facturas</p>
                  <p className="mt-1 font-semibold">{selectedSegment.n_facturas_total ?? 0}</p>
                </div>
                <div className="rounded-md border p-3">
                  <p className="text-xs uppercase text-muted-foreground">Cortes</p>
                  <p className="mt-1 font-semibold">{selectedSegment.n_cortes_total ?? 0}</p>
                </div>
              </div>
            </div>
          ) : (
            <Empty className="min-h-[320px] border-0">
              <EmptyHeader>
                <EmptyMedia variant="icon">
                  <UsersRound />
                </EmptyMedia>
                <EmptyTitle>Perfil del cliente</EmptyTitle>
                <EmptyDescription>Selecciona un cliente para ver su segmento.</EmptyDescription>
              </EmptyHeader>
            </Empty>
          )}
        </CardContent>
      </Card>
    </section>
  );
}
