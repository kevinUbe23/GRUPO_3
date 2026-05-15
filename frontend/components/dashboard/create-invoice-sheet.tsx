"use client";

import { FormEvent, useState } from "react";
import { FilePlus2, Loader2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Field,
  FieldDescription,
  FieldError,
  FieldGroup,
  FieldLabel
} from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue
} from "@/components/ui/select";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
  SheetTrigger
} from "@/components/ui/sheet";
import type { InvoiceCreateInput } from "@/lib/types";

type InvoiceState = InvoiceCreateInput["estado_factura"];

type CreateInvoiceSheetProps = {
  fechaCorte: string;
  disabled: boolean;
  onCreate: (payload: InvoiceCreateInput) => Promise<void>;
};

const invoiceStates: Array<{ value: InvoiceState; label: string }> = [
  { value: "abierta", label: "Abierta" },
  { value: "en_disputa", label: "En disputa" },
  { value: "pagada", label: "Pagada" },
  { value: "anulada", label: "Anulada" },
  { value: "castigada", label: "Castigada" }
];

export function CreateInvoiceSheet({ fechaCorte, disabled, onCreate }: CreateInvoiceSheetProps) {
  const [open, setOpen] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [estado, setEstado] = useState<InvoiceState>("abierta");

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    const form = new FormData(event.currentTarget);
    const facturaId = String(form.get("factura_id") ?? "").trim();
    const clienteId = String(form.get("cliente_id") ?? "").trim();
    const fechaEmision = String(form.get("fecha_emision") ?? "");
    const fechaVencimiento = String(form.get("fecha_vencimiento") ?? "");
    const fechaPagoReal = String(form.get("fecha_pago_real") ?? "");
    const monto = Number(form.get("monto"));
    const saldoRaw = String(form.get("saldo_pendiente") ?? "");

    if (!clienteId || !fechaEmision || !fechaVencimiento || !Number.isFinite(monto) || monto <= 0) {
      setError("Completa cliente, fechas y monto antes de crear la factura.");
      return;
    }
    if (estado === "pagada" && !fechaPagoReal) {
      setError("Una factura pagada necesita fecha de pago.");
      return;
    }

    setPending(true);
    try {
      await onCreate({
        ...(facturaId ? { factura_id: facturaId } : {}),
        cliente_id: clienteId,
        fecha_emision: fechaEmision,
        fecha_vencimiento: fechaVencimiento,
        fecha_pago_real: fechaPagoReal || null,
        monto,
        saldo_pendiente: saldoRaw ? Number(saldoRaw) : null,
        estado_factura: estado
      });
      event.currentTarget.reset();
      setEstado("abierta");
      setOpen(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo crear la factura.");
    } finally {
      setPending(false);
    }
  }

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <Button variant="outline" disabled={disabled}>
          <FilePlus2 data-icon="inline-start" />
          Nueva factura
        </Button>
      </SheetTrigger>
      <SheetContent className="sm:max-w-md">
        <SheetHeader>
          <SheetTitle>Nueva factura</SheetTitle>
          <SheetDescription>Registra una factura operativa sin cargar archivos externos.</SheetDescription>
        </SheetHeader>
        <form id="create-invoice-form" onSubmit={handleSubmit} className="overflow-auto px-4">
          <FieldGroup>
            <Field>
              <FieldLabel htmlFor="factura_id">Factura</FieldLabel>
              <Input id="factura_id" name="factura_id" placeholder="Autogenerada si se deja vacia" />
              <FieldDescription>Usa un codigo propio solo si ya existe en tu control operativo.</FieldDescription>
            </Field>
            <Field>
              <FieldLabel htmlFor="cliente_id">Cliente</FieldLabel>
              <Input id="cliente_id" name="cliente_id" placeholder="CLI000001" required />
            </Field>
            <div className="grid gap-3 sm:grid-cols-2">
              <Field>
                <FieldLabel htmlFor="fecha_emision">Emision</FieldLabel>
                <Input id="fecha_emision" name="fecha_emision" type="date" defaultValue={fechaCorte} required />
              </Field>
              <Field>
                <FieldLabel htmlFor="fecha_vencimiento">Vencimiento</FieldLabel>
                <Input id="fecha_vencimiento" name="fecha_vencimiento" type="date" defaultValue={fechaCorte} required />
              </Field>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <Field>
                <FieldLabel htmlFor="monto">Monto</FieldLabel>
                <Input id="monto" name="monto" type="number" min="0.01" step="0.01" required />
              </Field>
              <Field>
                <FieldLabel htmlFor="saldo_pendiente">Saldo pendiente</FieldLabel>
                <Input id="saldo_pendiente" name="saldo_pendiente" type="number" min="0" step="0.01" />
              </Field>
            </div>
            <Field>
              <FieldLabel>Estado</FieldLabel>
              <Select value={estado} onValueChange={(value) => setEstado(value as InvoiceState)}>
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    {invoiceStates.map((item) => (
                      <SelectItem key={item.value} value={item.value}>
                        {item.label}
                      </SelectItem>
                    ))}
                  </SelectGroup>
                </SelectContent>
              </Select>
            </Field>
            {estado === "pagada" && (
              <Field>
                <FieldLabel htmlFor="fecha_pago_real">Fecha de pago</FieldLabel>
                <Input id="fecha_pago_real" name="fecha_pago_real" type="date" defaultValue={fechaCorte} />
              </Field>
            )}
            {error && <FieldError>{error}</FieldError>}
          </FieldGroup>
        </form>
        <SheetFooter>
          <Button type="submit" form="create-invoice-form" disabled={pending || disabled}>
            {pending ? <Loader2 data-icon="inline-start" className="animate-spin" /> : <FilePlus2 data-icon="inline-start" />}
            Crear factura
          </Button>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  );
}
