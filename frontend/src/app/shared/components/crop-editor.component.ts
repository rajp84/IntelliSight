import { Component, EventEmitter, Input, Output } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-crop-editor',
  standalone: true,
  imports: [CommonModule],
  template: `
  <div class="space-y-3">
    <div class="relative w-full"
         (mousemove)="onPointerMove($event)"
         (mouseup)="endDrag()"
         (mouseleave)="endDrag()">
      <img [src]="imageUrl" alt="full" class="w-full bg-black" (load)="onImgLoad($event)" />
      <div *ngIf="hasBBox" class="absolute cursor-move"
           [ngStyle]="bboxStyle()"
           (mousedown)="startDrag($event)">
        <!-- Resize handles -->
        <div class="absolute w-3 h-3 bg-white border border-emerald-600 -top-2 left-1/2 -translate-x-1/2 cursor-n-resize" (mousedown)="startResize('n', $event)"></div>
        <div class="absolute w-3 h-3 bg-white border border-emerald-600 -bottom-2 left-1/2 -translate-x-1/2 cursor-s-resize" (mousedown)="startResize('s', $event)"></div>
        <div class="absolute w-3 h-3 bg-white border border-emerald-600 top-1/2 -right-2 -translate-y-1/2 cursor-e-resize" (mousedown)="startResize('e', $event)"></div>
        <div class="absolute w-3 h-3 bg-white border border-emerald-600 top-1/2 -left-2 -translate-y-1/2 cursor-w-resize" (mousedown)="startResize('w', $event)"></div>
        <div class="absolute w-3 h-3 bg-white border border-emerald-600 -top-2 -right-2 cursor-ne-resize" (mousedown)="startResize('ne', $event)"></div>
        <div class="absolute w-3 h-3 bg-white border border-emerald-600 -top-2 -left-2 cursor-nw-resize" (mousedown)="startResize('nw', $event)"></div>
        <div class="absolute w-3 h-3 bg-white border border-emerald-600 -bottom-2 -right-2 cursor-se-resize" (mousedown)="startResize('se', $event)"></div>
        <div class="absolute w-3 h-3 bg-white border border-emerald-600 -bottom-2 -left-2 cursor-sw-resize" (mousedown)="startResize('sw', $event)"></div>
      </div>
    </div>
  </div>
  `,
})
export class CropEditorComponent {
  @Input() imageUrl: string = '';
  @Input() bbox: { x1: number; y1: number; x2: number; y2: number } | null = null;
  @Output() bboxChange = new EventEmitter<{ x1: number; y1: number; x2: number; y2: number }>();

  natW = 0; natH = 0; renW = 0; renH = 0;
  dragging = false; resizing = false; handle: 'n'|'s'|'e'|'w'|'ne'|'nw'|'se'|'sw'|null = null;
  startRX = 0; startRY = 0; startBBox: { x1: number; y1: number; x2: number; y2: number } = { x1: 0, y1: 0, x2: 0, y2: 0 };
  containerRect: DOMRect | null = null;

  get hasBBox(): boolean {
    const b = this.bbox;
    return !!b && isFinite(b.x1) && isFinite(b.y1) && isFinite(b.x2) && isFinite(b.y2) && b.x2 > b.x1 && b.y2 > b.y1;
  }

  onImgLoad(ev: Event): void {
    const img = ev.target as HTMLImageElement;
    this.natW = Number(img.naturalWidth || 0);
    this.natH = Number(img.naturalHeight || 0);
    this.renW = Number(img.clientWidth || img.width || 0);
    this.renH = Number(img.clientHeight || img.height || 0);
    // Initialize a default bbox if not provided so users can see/adjust
    if (!this.hasBBox && this.natW > 0 && this.natH > 0) {
      const w = Math.round(this.natW * 0.3);
      const h = Math.round(this.natH * 0.3);
      const x1 = Math.max(0, Math.round((this.natW - w) / 2));
      const y1 = Math.max(0, Math.round((this.natH - h) / 2));
      const x2 = Math.min(this.natW, x1 + w);
      const y2 = Math.min(this.natH, y1 + h);
      this.setBBox(x1, y1, x2, y2);
    }
  }

  bboxStyle(): { [k: string]: string } {
    if (!this.hasBBox || this.natW <= 0 || this.natH <= 0 || this.renW <= 0 || this.renH <= 0) return {};
    const sx = this.renW / this.natW; const sy = this.renH / this.natH;
    const left = Math.max(0, Math.round((this.bbox!.x1) * sx));
    const top = Math.max(0, Math.round((this.bbox!.y1) * sy));
    const width = Math.max(1, Math.round((this.bbox!.x2 - this.bbox!.x1) * sx));
    const height = Math.max(1, Math.round((this.bbox!.y2 - this.bbox!.y1) * sy));
    return { left: `${left}px`, top: `${top}px`, width: `${width}px`, height: `${height}px`, border: '2px solid #22c55e', boxSizing: 'border-box' };
  }

  private clamp(v: number, lo: number, hi: number): number { return Math.max(lo, Math.min(hi, v)); }

  startDrag(e: MouseEvent): void {
    if (!this.hasBBox) return;
    e.preventDefault();
    const target = e.currentTarget as HTMLElement | null;
    const parent = target?.parentElement || null;
    const rect = parent?.getBoundingClientRect() || null;
    if (!rect) return;
    this.containerRect = rect;
    this.startRX = e.clientX - rect.left; this.startRY = e.clientY - rect.top;
    this.startBBox = { ...this.bbox! };
    this.dragging = true; this.resizing = false; this.handle = null;
  }

  startResize(h: 'n'|'s'|'e'|'w'|'ne'|'nw'|'se'|'sw', e: MouseEvent): void {
    e.preventDefault(); e.stopPropagation();
    const parent = (e.currentTarget as HTMLElement)?.parentElement?.parentElement || null;
    const rect = parent?.getBoundingClientRect() || null;
    if (!rect) return;
    this.containerRect = rect;
    this.startRX = e.clientX - rect.left; this.startRY = e.clientY - rect.top;
    this.startBBox = { ...this.bbox! };
    this.resizing = true; this.dragging = false; this.handle = h;
  }

  onPointerMove(e: MouseEvent): void {
    if (!this.containerRect || !this.bbox) return;
    e.preventDefault();
    const rx = e.clientX - this.containerRect.left; const ry = e.clientY - this.containerRect.top;
    const dRx = rx - this.startRX; const dRy = ry - this.startRY;
    const sx = this.renW > 0 ? (this.renW / this.natW) : 1; const sy = this.renH > 0 ? (this.renH / this.natH) : 1;
    const dNx = sx ? dRx / sx : 0; const dNy = sy ? dRy / sy : 0;
    const minW = 1; const minH = 1;
    if (this.dragging) {
      const w = this.startBBox.x2 - this.startBBox.x1; const h = this.startBBox.y2 - this.startBBox.y1;
      let nx1 = this.startBBox.x1 + dNx; let ny1 = this.startBBox.y1 + dNy;
      nx1 = this.clamp(nx1, 0, Math.max(0, this.natW - w)); ny1 = this.clamp(ny1, 0, Math.max(0, this.natH - h));
      const nx2 = nx1 + w; const ny2 = ny1 + h;
      this.setBBox(nx1, ny1, nx2, ny2);
      return;
    }
    if (this.resizing && this.handle) {
      let { x1, y1, x2, y2 } = this.startBBox;
      if (this.handle.includes('e')) x2 = this.clamp(x2 + dNx, x1 + minW, this.natW);
      if (this.handle.includes('w')) x1 = this.clamp(x1 + dNx, 0, x2 - minW);
      if (this.handle.includes('s')) y2 = this.clamp(y2 + dNy, y1 + minH, this.natH);
      if (this.handle.includes('n')) y1 = this.clamp(y1 + dNy, 0, y2 - minH);
      this.setBBox(x1, y1, x2, y2);
      return;
    }
  }

  endDrag(): void { this.dragging = false; this.resizing = false; this.handle = null; this.containerRect = null; }

  private setBBox(x1: number, y1: number, x2: number, y2: number): void {
    const nx1 = Math.round(x1), ny1 = Math.round(y1), nx2 = Math.round(x2), ny2 = Math.round(y2);
    this.bbox = { x1: nx1, y1: ny1, x2: nx2, y2: ny2 };
    this.bboxChange.emit(this.bbox);
  }
}


