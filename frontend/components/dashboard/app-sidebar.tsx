import {
  BarChart3,
  CheckCircle2,
  ClipboardList,
  Clock3,
  LayoutDashboard,
  Settings2,
  UsersRound
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Sheet, SheetContent, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { cn } from "@/lib/utils";

const navItems = [
  { value: "preventive", label: "Cola preventiva", icon: ClipboardList },
  { value: "overdue", label: "Vencidas", icon: Clock3 },
  { value: "paid", label: "Pagadas", icon: CheckCircle2 },
  { value: "customers", label: "Clientes", icon: UsersRound },
  { value: "metrics", label: "Métricas", icon: BarChart3 }
] as const;

export type DashboardView = (typeof navItems)[number]["value"];

type SidebarContentProps = {
  activeView: DashboardView;
  onViewChange: (view: DashboardView) => void;
};

function SidebarContent({ activeView, onViewChange }: SidebarContentProps) {
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
        {navItems.map((item) => (
          <button
            key={item.value}
            type="button"
            className={cn(
              "flex h-9 items-center gap-2 rounded-md px-3 text-left text-sm font-medium text-muted-foreground transition-colors hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
              activeView === item.value && "bg-sidebar-accent text-sidebar-accent-foreground"
            )}
            onClick={() => onViewChange(item.value)}
          >
            <item.icon />
            <span>{item.label}</span>
          </button>
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

type SidebarProps = {
  activeView: DashboardView;
  onViewChange: (view: DashboardView) => void;
};

export function DesktopSidebar({ activeView, onViewChange }: SidebarProps) {
  return (
    <aside className="fixed inset-y-0 left-0 hidden w-64 border-r bg-sidebar lg:block">
      <SidebarContent activeView={activeView} onViewChange={onViewChange} />
    </aside>
  );
}

type MobileSidebarProps = {
  className?: string;
  activeView: DashboardView;
  onViewChange: (view: DashboardView) => void;
};

export function MobileSidebar({ className, activeView, onViewChange }: MobileSidebarProps) {
  return (
    <Sheet>
      <SheetTrigger asChild>
        <Button variant="outline" size="icon" className={className} aria-label="Abrir menu">
          <LayoutDashboard data-icon="inline-start" />
        </Button>
      </SheetTrigger>
      <SheetContent side="left" className="w-72 p-0">
        <SheetTitle className="sr-only">Menu principal</SheetTitle>
        <SidebarContent activeView={activeView} onViewChange={onViewChange} />
      </SheetContent>
    </Sheet>
  );
}
