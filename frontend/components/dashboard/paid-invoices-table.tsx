"use client";

import { CheckCircle2, Search } from "lucide-react";

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
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { compactDate, money } from "@/lib/formatters";
import type { PrioritizedInvoice } from "@/lib/types";

type PaidInvoicesTableProps = {
  rows: PrioritizedInvoice[];
  query: string;
  onQueryChange: (value: string) => void;
  selected: PrioritizedInvoice | null;
  onSelect: (row: PrioritizedInvoice) => void;
};

export function PaidInvoicesTable({
  rows,
  query,
  onQueryChange,
  selected,
  onSelect
}: PaidInvoicesTableProps) {
  return (
    <Card className="min-w-0">
      <CardHeader className="gap-3 border-b p-4 md:flex-row md:items-center md:justify-between">
        <div className="flex items-center gap-2">
          <CheckCircle2 className="size-5 text-muted-foreground" />
          <CardTitle className="text-base">Pagadas al corte</CardTitle>
          <Badge variant="secondary">{rows.length.toLocaleString("es-EC")}</Badge>
        </div>
        <label className="relative block md:w-80">
          <Search className="pointer-events-none absolute left-2.5 top-2 size-4 text-muted-foreground" />
          <Input
            value={query}
            onChange={(event) => onQueryChange(event.target.value)}
            placeholder="Buscar factura, cliente, sector"
            aria-label="Buscar factura, cliente o sector"
            className="pl-8"
          />
        </label>
      </CardHeader>
      <CardContent className="p-0">
        <div className="max-h-[640px] overflow-auto scrollbar-thin">
          <Table className="min-w-[920px]">
            <TableHeader className="sticky top-0 z-10 bg-muted">
              <TableRow>
                <TableHead>Factura</TableHead>
                <TableHead>Cliente</TableHead>
                <TableHead>Monto</TableHead>
                <TableHead>Vencimiento</TableHead>
                <TableHead>Pago real</TableHead>
                <TableHead>Mora real</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((row) => (
                <TableRow
                  key={`${row.factura_id}-${row.fecha_corte}`}
                  onClick={() => onSelect(row)}
                  aria-selected={selected?.factura_id === row.factura_id}
                  className="cursor-pointer"
                >
                  <TableCell className="font-semibold">{row.factura_id}</TableCell>
                  <TableCell>
                    <div className="max-w-[260px] truncate font-medium">{row.cliente_nombre}</div>
                    <div className="text-xs text-muted-foreground">
                      {row.sector} / {row.cliente_id}
                    </div>
                  </TableCell>
                  <TableCell className="font-medium">{money.format(row.monto)}</TableCell>
                  <TableCell>{compactDate(row.fecha_vencimiento)}</TableCell>
                  <TableCell>{compactDate(row.fecha_pago_real)}</TableCell>
                  <TableCell>
                    <Badge variant="outline">{(row.dias_mora_real ?? 0).toLocaleString("es-EC")} dias</Badge>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          {rows.length === 0 && (
            <Empty className="min-h-[300px] border-0">
              <EmptyHeader>
                <EmptyMedia variant="icon">
                  <CheckCircle2 />
                </EmptyMedia>
                <EmptyTitle>Sin facturas pagadas</EmptyTitle>
                <EmptyDescription>No hay pagos reales visibles para la fecha de corte seleccionada.</EmptyDescription>
              </EmptyHeader>
            </Empty>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
