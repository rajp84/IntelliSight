import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

@Injectable({ providedIn: 'root' })
export class ContextMenuService {
  // Emits the id of the card whose context menu should be open, or null to close all
  readonly openMenuId$ = new BehaviorSubject<string | null>(null);

  open(id: string) {
    this.openMenuId$.next(id);
  }

  closeAll() {
    this.openMenuId$.next(null);
  }
}


