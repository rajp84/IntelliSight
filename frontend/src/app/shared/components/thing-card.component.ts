import { Component, EventEmitter, HostListener, Input, Output, inject, OnDestroy, OnInit } from '@angular/core';
import { Subscription } from 'rxjs';
import { ContextMenuService } from '../services/context-menu.service';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-thing-card',
  standalone: true,
  imports: [CommonModule],
  template: `
  <div class="border rounded bg-white shadow-sm overflow-hidden cursor-pointer transition-all duration-200"
       [class.ring-2]="selected"
       [class.ring-blue-500]="selected"
       [class.bg-blue-50]="selected"
       (click)="cardClick.emit()"
       (contextmenu)="onContextMenu($event)">
    <div class="relative">
      <img [src]="imageSrc" alt="thing" class="w-full aspect-square object-contain bg-black" loading="lazy"
           (error)="onImgError($event)" />
      <div *ngIf="selected"
           class="absolute top-1 right-1 w-4 h-4 bg-blue-600 rounded-full flex items-center justify-center">
        <svg class="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
          <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"></path>
        </svg>
      </div>
    </div>
    <div class="p-2 text-xs text-gray-700">
      <div class="font-mono text-[8px] text-gray-500 truncate">{{ id }}</div>
      <div *ngIf="showLabel" class="truncate text-[14px] font-medium">{{ label || '-' }}</div>
    </div>

    <!-- Context menu -->
    <div *ngIf="enableContextMenu && showMenu"
         class="fixed z-50 bg-white border border-gray-200 rounded shadow-md text-sm"
         [style.left.px]="menuX" [style.top.px]="menuY"
         (click)="$event.stopPropagation()">
      <button class="block w-full text-left px-3 py-2 hover:bg-gray-100" (click)="onSetThumbnail()">Set as thumbnail</button>
    </div>
  </div>
  `,
})
export class ThingCardComponent implements OnInit, OnDestroy {
  @Input() id: string = '';
  @Input() label: string = '';
  @Input() imageSrc: string = '';
  @Input() selected: boolean = false;
  @Input() showLabel: boolean = true;
  @Input() enableContextMenu: boolean = false;
  @Output() cardClick = new EventEmitter<void>();
  @Output() setThumbnail = new EventEmitter<void>();

  onImgError(evt: Event) {
    const img = evt.target as HTMLImageElement;
    if (img && this.imageSrc) {
      // Try force reload once by appending cache buster
      const u = new URL(this.imageSrc, window.location.origin);
      u.searchParams.set('t', Date.now().toString());
      img.src = u.pathname + u.search;
    }
  }

  private readonly ctxMenu = inject(ContextMenuService);
  private sub?: Subscription;
  showMenu = false;
  menuX = 0;
  menuY = 0;

  onContextMenu(event: MouseEvent) {
    if (!this.enableContextMenu) return;
    event.preventDefault();
    event.stopPropagation();
    this.menuX = event.clientX;
    this.menuY = event.clientY;
    this.ctxMenu.open(this.id);
  }

  onSetThumbnail() {
    this.showMenu = false;
    this.setThumbnail.emit();
  }

  @HostListener('document:click')
  onDocClick() {
    this.ctxMenu.closeAll();
  }

  ngOnInit(): void {
    this.sub = this.ctxMenu.openMenuId$.subscribe(openId => {
      this.showMenu = !!this.enableContextMenu && openId === this.id;
    });
  }

  ngOnDestroy(): void {
    try { this.sub?.unsubscribe(); } catch {}
  }
}


