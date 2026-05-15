import { BarChart3, ClipboardList, FileText, LayoutDashboard, Settings2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Sheet, SheetContent, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { cn } from "@/lib/utils";

const navItems = [
  { label: "Resumen", href: "#resumen", icon: LayoutDashboard },
  { label: "Cola priorizada", href: "#cola", icon: ClipboardList },
  { label: "Detalle", href: "#detalle", icon: FileText },
  { label: "Metricas", href: "#resumen", icon: BarChart3 }
];

function SidebarContent() {
  return (
    <div className="flex h-full flex-col bg-sidebar text-sidebar-foreground">
      <div className="px-4 py-5">
        <div className="flex items-center gap-3">
          <span className="flex size-9 items-center justify-center rounded-lg bg-primary text-primary-foreground">
            <Settings2 className="size-4" />
          </span>
          <div className="min-w-0">
            <p className="truncate text-sm font-semibold">Cobranzas IA</p>
            <p className="truncate text-xs text-muted-foreground">Priorizacion operativa</p>
          </div>
        </div>
      </div>
      <Separator />
      <nav className="flex flex-1 flex-col gap-1 px-3 py-4">
        {navItems.map((item, index) => (
          <a
            key={`${item.label}-${index}`}
            href={item.href}
            className={cn(
              "flex h-9 items-center gap-2 rounded-md px-3 text-sm font-medium text-muted-foreground transition-colors hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
              index === 0 && "bg-sidebar-accent text-sidebar-accent-foreground"
            )}
          >
            <item.icon className="size-4" />
            <span>{item.label}</span>
          </a>
        ))}
      </nav>
      <div className="border-t p-4">
        <p className="text-xs leading-5 text-muted-foreground">
          Unidad de negocio: factura. Split metodologico por `factura_id`.
        </p>
      </div>
    </div>
  );
}

export function DesktopSidebar() {
  return (
    <aside className="fixed inset-y-0 left-0 hidden w-64 border-r bg-sidebar lg:block">
      <SidebarContent />
    </aside>
  );
}

type MobileSidebarProps = {
  className?: string;
};

export function MobileSidebar({ className }: MobileSidebarProps) {
  return (
    <Sheet>
      <SheetTrigger asChild>
        <Button variant="outline" size="icon" className={className} aria-label="Abrir menu">
          <LayoutDashboard data-icon="inline-start" />
        </Button>
      </SheetTrigger>
      <SheetContent side="left" className="w-72 p-0">
        <SheetTitle className="sr-only">Menu principal</SheetTitle>
        <SidebarContent />
      </SheetContent>
    </Sheet>
  );
}
