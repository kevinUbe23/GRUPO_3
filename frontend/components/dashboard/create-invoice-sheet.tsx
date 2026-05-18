"use client";

import { FormEvent, useDeferredValue, useEffect, useState } from "react";
import { ChevronsUpDown, FilePlus2, Loader2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList
} from "@/components/ui/command";
import {
  Field,
  FieldDescription,
  FieldError,
  FieldGroup,
  FieldLabel
} from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
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
import { api } from "@/lib/api";
import type { Customer, InvoiceCreateInput } from "@/lib/types";
import { cn } from "@/lib/utils";

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

const customerOptionsId = "customer-options";

function customerLabel(customer: Customer) {
  return `${customer.nombre} (${customer.cliente_id})`;
}

export function CreateInvoiceSheet({ fechaCorte, disabled, onCreate }: CreateInvoiceSheetProps) {
  const [open, setOpen] = useState(false);
  const [customerPopoverOpen, setCustomerPopoverOpen] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [estado, setEstado] = useState<InvoiceState>("abierta");
  const [customerQuery, setCustomerQuery] = useState("");
  const [customerOptions, setCustomerOptions] = useState<Customer[]>([]);
  const [selectedCustomer, setSelectedCustomer] = useState<Customer | null>(null);
  const [customerLoading, setCustomerLoading] = useState(false);
  const [customerError, setCustomerError] = useState<string | null>(null);
  const deferredCustomerQuery = useDeferredValue(customerQuery);

  useEffect(() => {
    if (!open || !customerPopoverOpen) {
      return;
    }

    let cancelled = false;
    setCustomerLoading(true);
    setCustomerError(null);
    setCustomerOptions([]);

    api.customers(8, 0, deferredCustomerQuery)
      .then((customers) => {
        if (!cancelled) setCustomerOptions(customers);
      })
      .catch((err) => {
        if (!cancelled) {
          setCustomerOptions([]);
          setCustomerError(err instanceof Error ? err.message : "No se pudieron cargar clientes.");
        }
      })
      .finally(() => {
        if (!cancelled) setCustomerLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [customerPopoverOpen, deferredCustomerQuery, open]);

  function handleCustomerSearch(value: string) {
    setCustomerQuery(value);
  }

  function handleCustomerSelect(customer: Customer) {
    setSelectedCustomer(customer);
    setCustomerQuery("");
    setCustomerPopoverOpen(false);
    setCustomerError(null);
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    const formElement = event.currentTarget;
    const form = new FormData(formElement);
    const facturaId = String(form.get("factura_id") ?? "").trim();
    const clienteId = selectedCustomer?.cliente_id ?? String(form.get("cliente_id") ?? "").trim();
    const fechaEmision = String(form.get("fecha_emision") ?? "");
    const fechaVencimiento = String(form.get("fecha_vencimiento") ?? "");
    const fechaPagoReal = String(form.get("fecha_pago_real") ?? "");
    const monto = Number(form.get("monto"));

    if (!clienteId || !fechaEmision || !fechaVencimiento || !Number.isFinite(monto) || monto <= 0) {
      setError("Selecciona un cliente, completa fechas y monto antes de crear la factura.");
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
        estado_factura: estado
      });
      formElement.reset();
      setEstado("abierta");
      setCustomerQuery("");
      setCustomerOptions([]);
      setSelectedCustomer(null);
      setCustomerPopoverOpen(false);
      setOpen(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo crear la factura.");
    } finally {
      setPending(false);
    }
  }

  const customerInvalid = Boolean(error && !selectedCustomer);

  return (
    <Sheet
      open={open}
      onOpenChange={(nextOpen) => {
        setOpen(nextOpen);
        if (!nextOpen) {
          setCustomerPopoverOpen(false);
        }
      }}
    >
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
              <Input id="factura_id" name="factura_id" placeholder="Se genera automaticamente si se deja vacia" />
              <FieldDescription>Opcional. Al crearla, el sistema la selecciona y muestra el codigo generado.</FieldDescription>
            </Field>
            <Field data-invalid={customerInvalid}>
              <FieldLabel htmlFor="cliente_busqueda">Cliente</FieldLabel>
              <Popover open={customerPopoverOpen} onOpenChange={setCustomerPopoverOpen}>
                <PopoverTrigger asChild>
                  <Button
                    id="cliente_busqueda"
                    type="button"
                    variant="outline"
                    role="combobox"
                    aria-controls={customerOptionsId}
                    aria-expanded={customerPopoverOpen}
                    aria-invalid={customerInvalid}
                    className={cn("w-full justify-between", !selectedCustomer && "text-muted-foreground")}
                    disabled={pending}
                  >
                    <span className="truncate">
                      {selectedCustomer ? customerLabel(selectedCustomer) : "Buscar y seleccionar cliente"}
                    </span>
                    <ChevronsUpDown data-icon="inline-end" />
                  </Button>
                </PopoverTrigger>
                <PopoverContent align="start" className="w-[var(--radix-popover-trigger-width)] p-0">
                  <Command shouldFilter={false}>
                    <CommandInput
                      value={customerQuery}
                      onValueChange={handleCustomerSearch}
                      placeholder="Buscar por nombre del cliente"
                    />
                    <CommandList id={customerOptionsId} aria-label="Clientes disponibles">
                      {customerLoading && <CommandEmpty>Cargando clientes...</CommandEmpty>}
                      {!customerLoading && customerError && <CommandEmpty>{customerError}</CommandEmpty>}
                      {!customerLoading && !customerError && customerOptions.length === 0 && (
                        <CommandEmpty>Sin clientes para esta busqueda.</CommandEmpty>
                      )}
                      {!customerLoading && !customerError && customerOptions.length > 0 && (
                        <CommandGroup>
                          {customerOptions.map((customer) => {
                            const selected = selectedCustomer?.cliente_id === customer.cliente_id;
                            return (
                              <CommandItem
                                key={customer.cliente_id}
                                data-checked={selected}
                                value={`${customer.nombre} ${customer.cliente_id} ${customer.sector}`}
                                onSelect={() => handleCustomerSelect(customer)}
                              >
                                <span className="flex min-w-0 flex-1 flex-col gap-0.5">
                                  <span className="truncate font-medium">{customer.nombre}</span>
                                  <span className="truncate text-xs text-muted-foreground">
                                    {customer.cliente_id} / {customer.sector}
                                  </span>
                                </span>
                              </CommandItem>
                            );
                          })}
                        </CommandGroup>
                      )}
                    </CommandList>
                  </Command>
                </PopoverContent>
              </Popover>
              <input type="hidden" name="cliente_id" value={selectedCustomer?.cliente_id ?? ""} />
              <FieldDescription>
                {selectedCustomer
                  ? `${selectedCustomer.cliente_id} / ${selectedCustomer.sector}`
                  : "Escribe el nombre y selecciona un cliente de la lista."}
              </FieldDescription>
              {customerError && <FieldError>{customerError}</FieldError>}
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
            <Field>
              <FieldLabel htmlFor="monto">Monto</FieldLabel>
              <Input id="monto" name="monto" type="number" min="0.01" step="0.01" required />
            </Field>
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
