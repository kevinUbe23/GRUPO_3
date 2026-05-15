import { Star } from "lucide-react";

import { cn } from "@/lib/utils";

type StarsProps = {
  value: number | null;
};

export function Stars({ value }: StarsProps) {
  const count = value ?? 0;

  return (
    <div className="flex items-center gap-0.5" aria-label={`${count} estrellas`}>
      {Array.from({ length: 5 }).map((_, index) => (
        <Star
          key={index}
          className={cn(
            "size-3.5",
            index < count ? "fill-primary text-primary" : "text-muted-foreground/40"
          )}
        />
      ))}
    </div>
  );
}
