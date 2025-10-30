import { Component, OnInit, inject } from '@angular/core';
import { CommonModule, NgFor, NgIf } from '@angular/common';
import { ActivatedRoute, RouterLink } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { ResultService, ThingGroupDetail } from '../../core/services/result.service';
import { ThingCardComponent } from '../../shared/components/thing-card.component';
import { ToastrService } from 'ngx-toastr';

@Component({
  selector: 'app-thing-group',
  standalone: true,
  imports: [CommonModule, NgIf, NgFor, RouterLink, ThingCardComponent, FormsModule],
  template: `
  <div class="p-2">
    <div class="flex items-center justify-between mb-3">
      <h2 class="text-lg font-semibold text-gray-900">{{ group?.name }}</h2>
      <div class="flex items-center space-x-2">
        <button (click)="openRenameModal()" class="px-3 py-1 text-sm text-gray-600 hover:text-gray-800 border border-gray-300 rounded hover:bg-gray-50">
          Rename Group
        </button>
        <a routerLink="/things" class="text-sm text-blue-600 hover:underline">Back to Things</a>
      </div>
    </div>
    <div class="flex items-center justify-between mb-2">
      <div class="text-sm text-gray-500" *ngIf="loading">Loading group...</div>
      <div class="text-sm text-gray-600" *ngIf="!loading && group">{{ group.items.length }} {{ group.items.length == 1 ? 'item' : 'items' }}</div>
      <div class="text-sm text-gray-600" *ngIf="selected.size > 0">
        {{ selected.size }} selected
      </div>
    </div>
    <div class="mb-3" *ngIf="selected.size > 0">
      <button (click)="clearSelection()" class="px-3 py-1 text-sm text-gray-600 hover:text-gray-800">Clear</button>
      <button (click)="removeSelected()" class="ml-2 px-3 py-1 text-sm rounded bg-red-600 text-white hover:bg-red-700">Remove from Group</button>
    </div>
    <div *ngIf="!loading && group" class="space-y-3">
      
      <div class="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-1">
        <app-thing-card *ngFor="let item of group.items"
                        [id]="item.id"
                        [label]="item.payload.label || '-'"
                        [imageSrc]="getImageUrl(item)"
                        [selected]="selected.has(item.id)"
                        [showLabel]="false"
                        [enableContextMenu]="true"
                        (cardClick)="toggle(item.id)"
                        (setThumbnail)="setAsThumbnail(item)"></app-thing-card>
      </div>
    </div>
  </div>

  <!-- Rename Group Modal -->
  <div *ngIf="showRenameModal" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
    <div class="bg-white rounded-lg p-6 w-full max-w-md mx-4">
      <div class="flex items-center justify-between mb-4">
        <h3 class="text-lg font-semibold text-gray-900">Rename Group</h3>
        <button (click)="closeRenameModal()" class="text-gray-400 hover:text-gray-600">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
          </svg>
        </button>
      </div>
      
      <div class="mb-4">
        <label class="block text-sm font-medium text-gray-700 mb-2">Group Name</label>
        <input type="text" [(ngModel)]="renameText" placeholder="Enter new group name"
          class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          (keyup.enter)="saveRename()" />
      </div>
      
      <div class="flex justify-end space-x-3">
        <button (click)="closeRenameModal()" [disabled]="renaming"
          class="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-md disabled:opacity-50 disabled:cursor-not-allowed">
          Cancel
        </button>
        <button (click)="saveRename()" [disabled]="!renameText.trim() || renaming"
          class="px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-md disabled:opacity-50 disabled:cursor-not-allowed flex items-center">
          <svg *ngIf="renaming" class="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg"
            fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z">
            </path>
          </svg>
          {{ renaming ? 'Renaming...' : 'Rename Group' }}
        </button>
      </div>
    </div>
  </div>
  `,
})
export class ThingGroupComponent implements OnInit {
  private readonly route = inject(ActivatedRoute);
  private readonly resultService = inject(ResultService);
  private readonly toastr = inject(ToastrService);

  loading = false;
  group: ThingGroupDetail | null = null;
  selected = new Set<string>();
  
  // Rename functionality
  showRenameModal = false;
  renameText = '';
  renaming = false;

  ngOnInit(): void {
    const groupId = this.route.snapshot.paramMap.get('groupId');
    if (!groupId) return;
    this.loading = true;
    this.resultService.getThingGroup(groupId).subscribe({
      next: (detail) => {
        this.group = detail;
      },
      error: () => {
        this.group = null;
      },
      complete: () => {
        this.loading = false;
      }
    });
  }

  getImageUrl(item: any): string {
    return this.resultService.getThingsImageUrl(item);
  }

  toggle(id: string): void {
    if (this.selected.has(id)) this.selected.delete(id); else this.selected.add(id);
  }

  clearSelection(): void {
    this.selected.clear();
  }

  removeSelected(): void {
    if (!this.group || this.selected.size === 0) return;
    const ids = Array.from(this.selected);
    this.resultService.removeThingsFromGroup(this.group.id, ids).subscribe({
      next: (res) => {
        // Remove from local group.items and update count
        this.group!.items = this.group!.items.filter(it => !this.selected.has(it.id));
        this.group!.count = this.group!.items.length;
        this.selected.clear();
      },
      error: () => {},
    });
  }

  setAsThumbnail(item: any): void {
    if (!this.group) return;
    const imageId = item?.payload?.image_id;
    if (!imageId) return;
    // Optimistically set while updating backend
    this.resultService
      .getThingGroup(this.group.id)
      .subscribe({ complete: () => {} });
    // Call backend to update thumbnail
    fetch(`/api/things/groups/${encodeURIComponent(this.group.id)}/thumbnail`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_id: imageId }),
    }).then(async (r) => {
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();
      this.group!.thumbnail_b64 = data.thumbnail_b64 || this.group!.thumbnail_b64;
    }).catch(() => {});
  }

  // Rename functionality
  openRenameModal(): void {
    if (!this.group) return;
    this.renameText = this.group.name;
    this.showRenameModal = true;
  }

  closeRenameModal(): void {
    this.showRenameModal = false;
    this.renameText = '';
    this.renaming = false;
  }

  saveRename(): void {
    if (!this.group || !this.renameText.trim() || this.renaming) return;
    
    const newName = this.renameText.trim();
    if (newName === this.group.name) {
      this.closeRenameModal();
      return;
    }
    
    this.renaming = true;
    
    this.resultService.renameThingGroup(this.group.id, newName).subscribe({
      next: (response) => {
        // Update local group data
        this.group!.name = newName;
        
        // Update all items in the group with the new label
        this.group!.items = this.group!.items.map(item => ({
          ...item,
          payload: {
            ...item.payload,
            label: newName
          }
        }));
        
        this.toastr.success(`Group renamed to "${newName}". Updated ${response.updated} items.`);
        this.closeRenameModal();
      },
      error: (err) => {
        console.error('Failed to rename group:', err);
        this.toastr.error('Failed to rename group');
        this.renaming = false;
      }
    });
  }
}


