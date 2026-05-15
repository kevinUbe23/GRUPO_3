"use client";

import { BarChart3, Gauge, Search } from "lucide-react";

import { Stars } from "@/components/dashboard/stars";
import { MetricHelp } from "@/components/dashboard/metric-help";
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
import { Input } from "@/components/ui/input";
import {
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious
} from "@/components/ui/pagination";
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { compactDate, money, percent, priorityActionHint, priorityClass, priorityLevel } from "@/lib/formatters";
import type { PrioritizedInvoice } from "@/lib/types";
import { cn } from "@/lib/utils";

type PrioritizedTableProps = {
  rows: PrioritizedInvoice[];
  totalRows: number;
  query: string;
  scoreFilter: "todos" | "critico" | "alto" | "medio" | "bajo";
  currentPage: number;
  totalPages: number;
  pageSize: number;
  selected: PrioritizedInvoice | null;
  onQueryChange: (value: string) => void;
  onScoreFilterChange: (value: "todos" | "critico" | "alto" | "medio" | "bajo") => void;
  onPageChange: (page: number) => void;
  onPageSizeChange: (value: number) => void;
  onSelect: (row: PrioritizedInvoice) => void;
};

function visiblePages(currentPage: number, totalPages: number) {
  const start = Math.max(1, currentPage - 2);
  const end = Math.min(totalPages, start + 4);
  return Array.from({ length: end - start + 1 }, (_, index) => start + index);
}

function predictionClass(label: string) {
  if (label.toLowerCase().includes("dentro del plazo")) {
    return "border-primary/20 bg-primary/10 text-primary";
  }
  if (label.toLowerCase().includes("critico")) {
    return "border-destructive/20 bg-destructive/10 text-destructive";
  }
  return "border-border bg-background text-foreground";
}

export function PrioritizedTable({
  rows,
  totalRows,
  query,
  scoreFilter,
  currentPage,
  totalPages,
  pageSize,
  selected,
  onQueryChange,
  onScoreFilterChange,
  onPageChange,
  onPageSizeChange,
  onSelect
}: PrioritizedTableProps) {
  const hasRows = rows.length > 0;
  const pages = visiblePages(currentPage, totalPages);
  const canGoPrevious = currentPage > 1;
  const canGoNext = currentPage < totalPages && totalRows > 0;
  const scoreOptions = [
    { value: "todos", label: "Todos" },
    { value: "critico", label: "Critica" },
    { value: "alto", label: "Alta" },
    { value: "medio", label: "Media" },
    { value: "bajo", label: "Baja" }
  ] as const;
  const scoreDescription =
    "Indice operativo de 0 a 100 para ordenar la cola: 0-39 baja, 40-59 media, 60-79 alta y 80+ critica. No es una probabilidad; indica que tan pronto conviene gestionar la factura.";
  const lateDescription =
    "Probabilidad de que la factura no se pague dentro del plazo pactado. Suma atraso leve, alto y critico. La mora grave se revisa en el detalle y equivale a atraso alto + critico.";

  function handlePageClick(event: React.MouseEvent<HTMLAnchorElement>, page: number) {
    event.preventDefault();
    onPageChange(page);
  }

  function handleRowKeyDown(event: React.KeyboardEvent<HTMLTableRowElement>, row: PrioritizedInvoice) {
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    onSelect(row);
  }

  return (
    <Card id="cola" className="min-w-0">
      <CardHeader className="gap-3 border-b p-4 md:flex-row md:items-center md:justify-between">
        <div className="flex items-center gap-2">
          <BarChart3 className="size-5 text-muted-foreground" />
          <CardTitle className="text-base">Cola priorizada</CardTitle>
          <Badge variant="secondary">{totalRows.toLocaleString("es-EC")}</Badge>
        </div>
        <div className="flex flex-col gap-2 md:flex-row md:items-center">
          <div className="flex rounded-md border bg-muted/40 p-1" role="group" aria-label="Filtrar por score de prioridad">
            {scoreOptions.map((option) => (
              <Button
                key={option.value}
                type="button"
                variant={scoreFilter === option.value ? "default" : "ghost"}
                size="sm"
                className="h-7 px-2 text-xs"
                aria-pressed={scoreFilter === option.value}
                onClick={() => onScoreFilterChange(option.value)}
              >
                {option.label}
              </Button>
            ))}
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
          <Select value={String(pageSize)} onValueChange={(value) => onPageSizeChange(Number(value))}>
            <SelectTrigger className="w-full md:w-32">
              <SelectValue aria-label={`${pageSize} filas`} />
            </SelectTrigger>
            <SelectContent>
              <SelectGroup>
                {[10, 15, 25, 50].map((value) => (
                  <SelectItem key={value} value={String(value)}>
                    {value} filas
                  </SelectItem>
                ))}
              </SelectGroup>
            </SelectContent>
          </Select>
        </div>
      </CardHeader>

      <CardContent className="p-0">
        <div className="max-h-[640px] overflow-auto scrollbar-thin">
          <Table className="min-w-[1120px]">
            <TableHeader className="sticky top-0 z-10 bg-muted">
              <TableRow>
                <TableHead>Factura</TableHead>
                <TableHead>Cliente</TableHead>
                <TableHead>Monto</TableHead>
                <TableHead>Prediccion</TableHead>
                <TableHead>
                  <span className="inline-flex items-center gap-1">
                    Score de prioridad
                    <MetricHelp label="Score de prioridad" description={scoreDescription} />
                  </span>
                </TableHead>
                <TableHead>
                  <span className="inline-flex items-center gap-1">
                    Prob. atraso
                    <MetricHelp label="Probabilidad de atraso" description={lateDescription} />
                  </span>
                </TableHead>
                <TableHead>Rating</TableHead>
                <TableHead>Accion</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((row) => (
                <TableRow
                  key={`${row.factura_id}-${row.fecha_corte}`}
                  onClick={() => onSelect(row)}
                  onKeyDown={(event) => handleRowKeyDown(event, row)}
                  tabIndex={0}
                  aria-selected={selected?.factura_id === row.factura_id}
                  className={cn(
                    "cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                    selected?.factura_id === row.factura_id && "bg-accent hover:bg-accent"
                  )}
                >
                  <TableCell className="align-top">
                    <div className="font-semibold text-foreground">{row.factura_id}</div>
                    <div className="text-xs text-muted-foreground">Vence {compactDate(row.fecha_vencimiento)}</div>
                  </TableCell>
                  <TableCell className="align-top">
                    <div className="max-w-[220px] truncate font-medium text-foreground">{row.cliente_nombre}</div>
                    <div className="text-xs text-muted-foreground">
                      {row.sector} / {row.cliente_id}
                    </div>
                  </TableCell>
                  <TableCell className="align-top font-medium">{money.format(row.monto)}</TableCell>
                  <TableCell className="align-top">
                    <Badge variant="outline" className={cn("max-w-[190px] justify-start truncate", predictionClass(row.predicted_label_usuario))}>
                      {row.predicted_label_usuario}
                    </Badge>
                  </TableCell>
                  <TableCell className="align-top">
                    <Badge variant="outline" className={cn("font-semibold", priorityClass(row.priority_score_0_100))}>
                      {row.priority_score_0_100.toFixed(1)}
                    </Badge>
                    <div className="mt-1 text-xs text-muted-foreground">
                      {priorityLevel(row.priority_score_0_100)} / {priorityActionHint(row.priority_score_0_100)}
                    </div>
                  </TableCell>
                  <TableCell className="align-top">
                    <div className="font-medium">{percent.format(row.any_late_probability)}</div>
                  </TableCell>
                  <TableCell className="align-top">
                    <Stars value={row.rating_estrellas} />
                  </TableCell>
                  <TableCell className="align-top">
                    <div className="max-w-[220px] truncate font-medium">{row.accion_sugerida}</div>
                    <div className="text-xs text-muted-foreground">Corte {compactDate(row.fecha_corte)}</div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>

          {!hasRows && (
            <Empty className="min-h-[300px] border-0">
              <EmptyHeader>
                <EmptyMedia variant="icon">
                  <Gauge />
                </EmptyMedia>
                <EmptyTitle>Sin cartera priorizada</EmptyTitle>
                <EmptyDescription>
                  Inicializa la base y recalcula la cartera para la fecha seleccionada.
                </EmptyDescription>
              </EmptyHeader>
            </Empty>
          )}
        </div>

        <div className="flex flex-col gap-3 border-t p-3 md:flex-row md:items-center md:justify-between">
          <p className="text-sm text-muted-foreground">
            Pagina {totalRows ? currentPage : 0} de {totalPages} / {totalRows.toLocaleString("es-EC")} resultados
          </p>
          <Pagination className="mx-0 w-auto justify-start md:justify-end">
            <PaginationContent>
              <PaginationItem>
                <PaginationPrevious
                  href="#"
                  text="Anterior"
                  aria-disabled={!canGoPrevious}
                  tabIndex={canGoPrevious ? 0 : -1}
                  className={cn(!canGoPrevious && "pointer-events-none opacity-50")}
                  onClick={(event) => {
                    if (!canGoPrevious) {
                      event.preventDefault();
                      return;
                    }
                    handlePageClick(event, currentPage - 1);
                  }}
                />
              </PaginationItem>
              {pages.map((page) => (
                <PaginationItem key={page}>
                  <PaginationLink
                    href="#"
                    isActive={page === currentPage}
                    onClick={(event) => handlePageClick(event, page)}
                  >
                    {page}
                  </PaginationLink>
                </PaginationItem>
              ))}
              <PaginationItem>
                <PaginationNext
                  href="#"
                  text="Siguiente"
                  aria-disabled={!canGoNext}
                  tabIndex={canGoNext ? 0 : -1}
                  className={cn(!canGoNext && "pointer-events-none opacity-50")}
                  onClick={(event) => {
                    if (!canGoNext) {
                      event.preventDefault();
                      return;
                    }
                    handlePageClick(event, currentPage + 1);
                  }}
                />
              </PaginationItem>
            </PaginationContent>
          </Pagination>
        </div>
      </CardContent>
    </Card>
  );
}
