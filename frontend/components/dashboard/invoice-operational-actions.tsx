"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import { ClipboardPlus, CreditCard, FilePenLine, Loader2, ReceiptText, ShieldAlert, ShieldCheck } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Field, FieldError, FieldGroup, FieldLabel } from "@/components/ui/field";
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
  SheetFooter,
  SheetHeader,
  SheetTitle,
  SheetTrigger
} from "@/components/ui/sheet";
import { Textarea } from "@/components/ui/textarea";
import type {
  Interaction,
  InteractionCreateInput,
  Invoice,
  InvoiceUpdateInput,
  PaymentCreateInput,
  PaymentPromise,
  PromiseCreateInput,
  PromiseUpdateInput,
  PrioritizedInvoice
} from "@/lib/types";

type InvoiceOperationalActionsProps = {
  selected: PrioritizedInvoice;
  invoice: Invoice | null;
  interactions: Interaction[];
  promises: PaymentPromise[];
  fechaCorte: string;
  disabled: boolean;
  onUpdateInvoice: (facturaId: string, payload: InvoiceUpdateInput) => Promise<void>;
  onCreateInteraction: (payload: InteractionCreateInput) => Promise<void>;
  onCreatePromise: (payload: PromiseCreateInput) => Promise<void>;
  onUpdatePromise: (promesaId: string, payload: PromiseUpdateInput) => Promise<void>;
  onRegisterPayment: (payload: PaymentCreateInput) => Promise<void>;
};

const channels = [
  { value: "llamada", label: "Llamada" },
  { value: "whatsapp", label: "WhatsApp" },
  { value: "email", label: "Correo electronico" },
  { value: "visita", label: "Visita" },
  { value: "carta_notarial", label: "Carta notarial" }
] as const;

const contactResults = [
  { value: "promesa_de_pago", label: "Promesa de pago" },
  { value: "confirma_pago", label: "Confirma intencion de pago" },
  { value: "en_proceso_interno", label: "En proceso interno" },
  { value: "disputa_monto", label: "Disputa por monto" },
  { value: "rechazo_pago", label: "Rechazo de pago" },
  { value: "pagado", label: "Pago reportado" }
];

const noContactResults = [
  { value: "no_contesta", label: "No contesta" },
  { value: "numero_invalido", label: "Numero invalido" },
  { value: "cliente_ausente", label: "Cliente ausente" }
];

const noPaymentReasons = [
  { value: "none", label: "Sin motivo" },
  { value: "flujo_caja", label: "Flujo de caja" },
  { value: "disputa_monto", label: "Disputa sobre el monto" },
  { value: "problema_facturacion", label: "Problema de facturacion" },
  { value: "proceso_interno", label: "Proceso interno pendiente" },
  { value: "no_recibio_factura", label: "No recibio factura" },
  { value: "sin_respuesta", label: "Sin respuesta" },
  { value: "otro", label: "Otro" }
];

export function InvoiceOperationalActions({
  selected,
  invoice,
  interactions,
  promises,
  fechaCorte,
  disabled,
  onUpdateInvoice,
  onCreateInteraction,
  onCreatePromise,
  onUpdatePromise,
  onRegisterPayment
}: InvoiceOperationalActionsProps) {
  const promiseInteractions = useMemo(
    () => interactions.filter((item) => item.resultado === "promesa_de_pago"),
    [interactions]
  );

  return (
    <div className="flex flex-wrap gap-2">
      <InvoiceEditSheet
        selected={selected}
        invoice={invoice}
        disabled={disabled}
        onUpdate={onUpdateInvoice}
      />
      <Button
        variant="outline"
        size="sm"
        disabled={disabled || selected.estado_factura_actual === "pagada"}
        onClick={() => onUpdateInvoice(selected.factura_id, { estado_factura: "en_disputa" })}
      >
        <ShieldAlert data-icon="inline-start" />
        Abrir disputa
      </Button>
      {selected.estado_factura_actual === "en_disputa" && (
        <Button
          variant="outline"
          size="sm"
          disabled={disabled}
          onClick={() => onUpdateInvoice(selected.factura_id, { estado_factura: "abierta" })}
        >
          <ShieldCheck data-icon="inline-start" />
          Cerrar disputa
        </Button>
      )}
      <InteractionSheet
        facturaId={selected.factura_id}
        fechaCorte={fechaCorte}
        fechaVencimiento={invoice?.fecha_vencimiento ?? selected.fecha_vencimiento}
        disabled={disabled}
        onCreate={onCreateInteraction}
      />
      <PromiseSheet
        interactions={promiseInteractions}
        fechaCorte={fechaCorte}
        disabled={disabled}
        onCreate={onCreatePromise}
      />
      <PaymentSheet
        facturaId={selected.factura_id}
        fechaCorte={fechaCorte}
        disabled={disabled || selected.estado_corte === "paid"}
        onCreate={onRegisterPayment}
      />
      <PromiseStatusSheet promises={promises} disabled={disabled || promises.length === 0} onUpdate={onUpdatePromise} />
    </div>
  );
}

function InvoiceEditSheet({
  selected,
  invoice,
  disabled,
  onUpdate
}: {
  selected: PrioritizedInvoice;
  invoice: Invoice | null;
  disabled: boolean;
  onUpdate: (facturaId: string, payload: InvoiceUpdateInput) => Promise<void>;
}) {
  const [open, setOpen] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    const formElement = event.currentTarget;
    const form = new FormData(formElement);
    const monto = Number(form.get("monto"));
    const saldo = Number(form.get("saldo_pendiente"));
    const payload: InvoiceUpdateInput = {
      fecha_emision: String(form.get("fecha_emision") ?? ""),
      fecha_vencimiento: String(form.get("fecha_vencimiento") ?? ""),
      monto,
      saldo_pendiente: saldo,
      estado_factura: String(form.get("estado_factura") ?? "abierta") as InvoiceUpdateInput["estado_factura"]
    };
    if (!payload.fecha_emision || !payload.fecha_vencimiento || !Number.isFinite(monto) || !Number.isFinite(saldo)) {
      setError("Completa fechas, monto y saldo.");
      return;
    }
    setPending(true);
    try {
      await onUpdate(selected.factura_id, payload);
      setOpen(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo editar la factura.");
    } finally {
      setPending(false);
    }
  }

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <Button variant="outline" size="sm" disabled={disabled}>
          <FilePenLine data-icon="inline-start" />
          Editar
        </Button>
      </SheetTrigger>
      <SheetContent className="sm:max-w-md">
        <SheetHeader>
          <SheetTitle>Editar factura</SheetTitle>
        </SheetHeader>
        <form id="invoice-edit-form" onSubmit={handleSubmit} className="overflow-auto px-4">
          <FieldGroup>
            <Field>
              <FieldLabel htmlFor="fecha_emision_edit">Fecha de emision</FieldLabel>
              <Input id="fecha_emision_edit" name="fecha_emision" type="date" defaultValue={invoice?.fecha_emision ?? selected.fecha_emision} required />
            </Field>
            <Field>
              <FieldLabel htmlFor="fecha_vencimiento_edit">Fecha de vencimiento</FieldLabel>
              <Input id="fecha_vencimiento_edit" name="fecha_vencimiento" type="date" defaultValue={invoice?.fecha_vencimiento ?? selected.fecha_vencimiento} required />
            </Field>
            <Field>
              <FieldLabel htmlFor="monto_edit">Monto</FieldLabel>
              <Input id="monto_edit" name="monto" type="number" min="0.01" step="0.01" defaultValue={invoice?.monto ?? selected.monto} required />
            </Field>
            <Field>
              <FieldLabel htmlFor="saldo_edit">Saldo pendiente</FieldLabel>
              <Input id="saldo_edit" name="saldo_pendiente" type="number" min="0" step="0.01" defaultValue={invoice?.saldo_pendiente ?? selected.monto} required />
            </Field>
            <Field>
              <FieldLabel>Estado</FieldLabel>
              <Select name="estado_factura" defaultValue={invoice?.estado_factura ?? selected.estado_factura_actual}>
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    <SelectItem value="abierta">Abierta</SelectItem>
                    <SelectItem value="en_disputa">En disputa</SelectItem>
                    <SelectItem value="anulada">Anulada</SelectItem>
                    <SelectItem value="castigada">Castigada</SelectItem>
                  </SelectGroup>
                </SelectContent>
              </Select>
            </Field>
            {error && <FieldError>{error}</FieldError>}
          </FieldGroup>
        </form>
        <SheetFooter>
          <Button type="submit" form="invoice-edit-form" disabled={pending || disabled}>
            {pending ? <Loader2 data-icon="inline-start" className="animate-spin" /> : <FilePenLine data-icon="inline-start" />}
            Guardar cambios
          </Button>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  );
}

function InteractionSheet({
  facturaId,
  fechaCorte,
  fechaVencimiento,
  disabled,
  onCreate
}: {
  facturaId: string;
  fechaCorte: string;
  fechaVencimiento: string;
  disabled: boolean;
  onCreate: (payload: InteractionCreateInput) => Promise<void>;
}) {
  const [open, setOpen] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [contacto, setContacto] = useState("true");
  const [resultado, setResultado] = useState("promesa_de_pago");
  const [motivo, setMotivo] = useState("none");
  const [fechaGestion, setFechaGestion] = useState(fechaCorte);
  const results = contacto === "true" ? contactResults : noContactResults;
  const shouldShowReason = contacto === "true" && fechaGestion > fechaVencimiento && resultado !== "pagado";

  useEffect(() => {
    setFechaGestion(fechaCorte);
  }, [facturaId, fechaCorte]);

  useEffect(() => {
    if (!shouldShowReason) setMotivo("none");
  }, [shouldShowReason]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    const formElement = event.currentTarget;
    const form = new FormData(formElement);
    const canal = String(form.get("canal") ?? "llamada") as InteractionCreateInput["canal"];
    const observacion = String(form.get("observacion") ?? "").trim();

    if (!fechaGestion || !resultado) {
      setError("Completa fecha y resultado de la gestion.");
      return;
    }

    setPending(true);
    try {
      await onCreate({
        factura_id: facturaId,
        fecha_gestion: fechaGestion,
        canal,
        contacto_exitoso: contacto === "true",
        resultado,
        motivo_no_pago: shouldShowReason && motivo !== "none" ? motivo : null,
        observacion: observacion || null,
        recalculate: false
      });
      formElement.reset();
      setContacto("true");
      setResultado("promesa_de_pago");
      setMotivo("none");
      setFechaGestion(fechaCorte);
      setOpen(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo registrar la gestion.");
    } finally {
      setPending(false);
    }
  }

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <Button variant="outline" size="sm" disabled={disabled}>
          <ClipboardPlus data-icon="inline-start" />
          Gestion
        </Button>
      </SheetTrigger>
      <SheetContent className="sm:max-w-md">
        <SheetHeader>
          <SheetTitle>Registrar gestion</SheetTitle>
        </SheetHeader>
        <form id="interaction-form" onSubmit={handleSubmit} className="overflow-auto px-4">
          <FieldGroup>
            <Field>
              <FieldLabel htmlFor="fecha_gestion">Fecha</FieldLabel>
              <Input
                id="fecha_gestion"
                name="fecha_gestion"
                type="date"
                value={fechaGestion}
                onChange={(event) => setFechaGestion(event.target.value)}
                required
              />
            </Field>
            <Field>
              <FieldLabel>Canal</FieldLabel>
              <Select name="canal" defaultValue="llamada">
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    {channels.map((item) => (
                      <SelectItem key={item.value} value={item.value}>
                        {item.label}
                      </SelectItem>
                    ))}
                  </SelectGroup>
                </SelectContent>
              </Select>
            </Field>
            <Field>
              <FieldLabel>Contacto</FieldLabel>
              <Select
                value={contacto}
                onValueChange={(value) => {
                  setContacto(value);
                  setResultado(value === "true" ? "promesa_de_pago" : "no_contesta");
                  setMotivo("none");
                }}
              >
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    <SelectItem value="true">Contacto efectivo</SelectItem>
                    <SelectItem value="false">Sin contacto</SelectItem>
                  </SelectGroup>
                </SelectContent>
              </Select>
            </Field>
            <Field>
              <FieldLabel>Resultado</FieldLabel>
              <Select
                value={resultado}
                onValueChange={(value) => {
                  setResultado(value);
                  setMotivo("none");
                }}
              >
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    {results.map((item) => (
                      <SelectItem key={item.value} value={item.value}>
                        {item.label}
                      </SelectItem>
                    ))}
                  </SelectGroup>
                </SelectContent>
              </Select>
            </Field>
            {shouldShowReason && (
              <Field>
                <FieldLabel>Motivo de no pago</FieldLabel>
                <Select value={motivo} onValueChange={setMotivo}>
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectGroup>
                      {noPaymentReasons.map((item) => (
                        <SelectItem key={item.value} value={item.value}>
                          {item.label}
                        </SelectItem>
                      ))}
                    </SelectGroup>
                  </SelectContent>
                </Select>
              </Field>
            )}
            <Field>
              <FieldLabel htmlFor="observacion">Observacion</FieldLabel>
              <Textarea id="observacion" name="observacion" rows={3} />
            </Field>
            {error && <FieldError>{error}</FieldError>}
          </FieldGroup>
        </form>
        <SheetFooter>
          <Button type="submit" form="interaction-form" disabled={pending || disabled}>
            {pending ? <Loader2 data-icon="inline-start" className="animate-spin" /> : <ClipboardPlus data-icon="inline-start" />}
            Guardar gestion
          </Button>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  );
}

function PromiseSheet({
  interactions,
  fechaCorte,
  disabled,
  onCreate
}: {
  interactions: Interaction[];
  fechaCorte: string;
  disabled: boolean;
  onCreate: (payload: PromiseCreateInput) => Promise<void>;
}) {
  const [open, setOpen] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [gestionId, setGestionId] = useState(interactions[0]?.gestion_id ?? "");

  useEffect(() => {
    setGestionId(interactions[0]?.gestion_id ?? "");
  }, [interactions]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    const formElement = event.currentTarget;
    const form = new FormData(formElement);
    const fechaCompromiso = String(form.get("fecha_compromiso") ?? "");
    if (!gestionId || !fechaCompromiso) {
      setError("Selecciona una gestion con promesa y una fecha de compromiso.");
      return;
    }

    setPending(true);
    try {
      await onCreate({ gestion_id: gestionId, fecha_compromiso: fechaCompromiso, recalculate: false });
      formElement.reset();
      setOpen(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo registrar la promesa.");
    } finally {
      setPending(false);
    }
  }

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <Button variant="outline" size="sm" disabled={disabled || interactions.length === 0}>
          <ReceiptText data-icon="inline-start" />
          Promesa
        </Button>
      </SheetTrigger>
      <SheetContent className="sm:max-w-md">
        <SheetHeader>
          <SheetTitle>Registrar promesa</SheetTitle>
        </SheetHeader>
        <form id="promise-form" onSubmit={handleSubmit} className="overflow-auto px-4">
          <FieldGroup>
            <Field>
              <FieldLabel>Gestion origen</FieldLabel>
              <Select value={gestionId} onValueChange={setGestionId}>
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Selecciona gestion" />
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    {interactions.map((item) => (
                      <SelectItem key={item.gestion_id} value={item.gestion_id}>
                        {item.gestion_id} / {item.fecha_gestion}
                      </SelectItem>
                    ))}
                  </SelectGroup>
                </SelectContent>
              </Select>
            </Field>
            <Field>
              <FieldLabel htmlFor="fecha_compromiso">Fecha compromiso</FieldLabel>
              <Input id="fecha_compromiso" name="fecha_compromiso" type="date" defaultValue={fechaCorte} required />
            </Field>
            {error && <FieldError>{error}</FieldError>}
          </FieldGroup>
        </form>
        <SheetFooter>
          <Button type="submit" form="promise-form" disabled={pending || disabled || interactions.length === 0}>
            {pending ? <Loader2 data-icon="inline-start" className="animate-spin" /> : <ReceiptText data-icon="inline-start" />}
            Guardar promesa
          </Button>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  );
}

function PaymentSheet({
  facturaId,
  fechaCorte,
  disabled,
  onCreate
}: {
  facturaId: string;
  fechaCorte: string;
  disabled: boolean;
  onCreate: (payload: PaymentCreateInput) => Promise<void>;
}) {
  const [open, setOpen] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    const formElement = event.currentTarget;
    const form = new FormData(formElement);
    const fechaPago = String(form.get("fecha_pago") ?? "");
    if (!fechaPago) {
      setError("Selecciona la fecha de pago.");
      return;
    }

    setPending(true);
    try {
      await onCreate({ factura_id: facturaId, fecha_pago: fechaPago });
      formElement.reset();
      setOpen(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo registrar el pago.");
    } finally {
      setPending(false);
    }
  }

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <Button variant="outline" size="sm" disabled={disabled}>
          <CreditCard data-icon="inline-start" />
          Pago
        </Button>
      </SheetTrigger>
      <SheetContent className="sm:max-w-md">
        <SheetHeader>
          <SheetTitle>Registrar pago</SheetTitle>
        </SheetHeader>
        <form id="payment-form" onSubmit={handleSubmit} className="overflow-auto px-4">
          <FieldGroup>
            <Field>
              <FieldLabel htmlFor="fecha_pago">Fecha de pago</FieldLabel>
              <Input id="fecha_pago" name="fecha_pago" type="date" defaultValue={fechaCorte} required />
            </Field>
            {error && <FieldError>{error}</FieldError>}
          </FieldGroup>
        </form>
        <SheetFooter>
          <Button type="submit" form="payment-form" disabled={pending || disabled}>
            {pending ? <Loader2 data-icon="inline-start" className="animate-spin" /> : <CreditCard data-icon="inline-start" />}
            Guardar pago
          </Button>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  );
}

function PromiseStatusSheet({
  promises,
  disabled,
  onUpdate
}: {
  promises: PaymentPromise[];
  disabled: boolean;
  onUpdate: (promesaId: string, payload: PromiseUpdateInput) => Promise<void>;
}) {
  const [open, setOpen] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [promesaId, setPromesaId] = useState(promises[0]?.promesa_id ?? "");
  const [estado, setEstado] = useState<PromiseUpdateInput["estado_promesa"]>("cumplida");

  useEffect(() => {
    setPromesaId(promises[0]?.promesa_id ?? "");
  }, [promises]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    if (!promesaId) {
      setError("Selecciona una promesa.");
      return;
    }
    setPending(true);
    try {
      await onUpdate(promesaId, { estado_promesa: estado, se_cumplio: estado === "cumplida" });
      setOpen(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo actualizar la promesa.");
    } finally {
      setPending(false);
    }
  }

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <Button variant="outline" size="sm" disabled={disabled}>
          <ShieldCheck data-icon="inline-start" />
          Estado promesa
        </Button>
      </SheetTrigger>
      <SheetContent className="sm:max-w-md">
        <SheetHeader>
          <SheetTitle>Actualizar promesa</SheetTitle>
        </SheetHeader>
        <form id="promise-status-form" onSubmit={handleSubmit} className="overflow-auto px-4">
          <FieldGroup>
            <Field>
              <FieldLabel>Promesa</FieldLabel>
              <Select value={promesaId} onValueChange={setPromesaId}>
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Selecciona promesa" />
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    {promises.map((item) => (
                      <SelectItem key={item.promesa_id} value={item.promesa_id}>
                        {item.promesa_id} / {item.fecha_compromiso} / {item.estado_promesa}
                      </SelectItem>
                    ))}
                  </SelectGroup>
                </SelectContent>
              </Select>
            </Field>
            <Field>
              <FieldLabel>Nuevo estado</FieldLabel>
              <Select value={estado} onValueChange={(value) => setEstado(value as PromiseUpdateInput["estado_promesa"])}>
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    <SelectItem value="cumplida">Cumplida</SelectItem>
                    <SelectItem value="activa">Activa</SelectItem>
                    <SelectItem value="incumplida">Incumplida</SelectItem>
                    <SelectItem value="cancelada">Cancelada</SelectItem>
                    <SelectItem value="reemplazada">Reemplazada</SelectItem>
                  </SelectGroup>
                </SelectContent>
              </Select>
            </Field>
            {error && <FieldError>{error}</FieldError>}
          </FieldGroup>
        </form>
        <SheetFooter>
          <Button type="submit" form="promise-status-form" disabled={pending || disabled}>
            {pending ? <Loader2 data-icon="inline-start" className="animate-spin" /> : <ShieldCheck data-icon="inline-start" />}
            Actualizar
          </Button>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  );
}
