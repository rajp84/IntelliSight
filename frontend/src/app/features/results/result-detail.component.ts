import { Component, OnInit, OnDestroy, AfterViewInit, inject } from '@angular/core';
import { CommonModule, NgFor, NgIf } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute } from '@angular/router';
import { ResultService, EmbeddingItem } from '../../core/services/result.service';
import { ToastrService } from 'ngx-toastr';

@Component({
  selector: 'app-result-detail',
  standalone: true,
  imports: [CommonModule, NgIf, NgFor, FormsModule],
  template: `
  <div class="p-4">
    <div class="text-sm text-gray-700 mb-3">Training Run: <span class="font-mono">{{ runId }}</span></div>
    <div *ngIf="loading" class="text-sm text-gray-500">Loading embeddings…</div>
    <div *ngIf="!loading && groupsKeys.length === 0" class="text-sm text-gray-500">No embeddings found.</div>
    <div *ngIf="!loading && groupsKeys.length > 0" class="mb-2 text-sm text-gray-600">Total loaded: {{ items.length }} detections</div>
    
    <!-- Sticky selection controls container -->
    <div *ngIf="!loading && groupsKeys.length > 0" 
         class="sticky top-0 z-30 bg-white border-b border-gray-200 shadow-sm mb-4">
      <div class="px-4 py-3 flex items-center justify-between">
        <div class="text-sm text-gray-600">
          <span *ngIf="selectedItems.size === 0">Click cards to select</span>
          <span *ngIf="selectedItems.size > 0">{{ selectedItems.size }} item(s) selected</span>
        </div>
        <div class="flex items-center space-x-2">
          <button *ngIf="selectedItems.size > 0" 
                  (click)="clearSelection()" 
                  [disabled]="committingToThings"
                  class="px-3 py-1 text-sm text-gray-600 hover:text-gray-800 disabled:opacity-50 disabled:cursor-not-allowed">
            Clear Selection
          </button>
          <button *ngIf="selectedItems.size > 0" 
                  (click)="openEditLabelModal()" 
                  [disabled]="committingToThings"
                  class="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm disabled:opacity-50 disabled:cursor-not-allowed">
            Edit Label
          </button>
          <button *ngIf="selectedItems.size > 0" 
                  (click)="commitToMyThings()" 
                  [disabled]="committingToThings"
                  class="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 text-sm disabled:opacity-50 disabled:cursor-not-allowed flex items-center">
            <svg *ngIf="committingToThings" class="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            {{ committingToThings ? 'Committing...' : 'Commit to My Things' }}
          </button>
        </div>
      </div>
    </div>
    
    <div *ngFor="let key of groupsKeys" class="mb-6">
      <div class="text-sm font-semibold text-gray-800 mb-2">{{ key }} ({{ groups[key].length }} detections)</div>
      <div class="grid grid-cols-6 md:grid-cols-8 lg:grid-cols-12 gap-1">
        <div *ngFor="let item of groups[key]" 
             class="border rounded bg-white shadow-sm overflow-hidden cursor-pointer transition-all duration-200"
             [class.ring-2]="selectedItems.has(item.id)"
             [class.ring-blue-500]="selectedItems.has(item.id)"
             [class.bg-blue-50]="selectedItems.has(item.id)"
             (click)="toggleSelection(item.id)">
          <div class="relative">
            <img [src]="getImageUrl(item)" alt="crop" class="w-full aspect-square object-contain bg-black" loading="lazy" />
            <div *ngIf="selectedItems.has(item.id)" 
                 class="absolute top-1 right-1 w-4 h-4 bg-blue-600 rounded-full flex items-center justify-center">
              <svg class="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"></path>
              </svg>
            </div>
          </div>
          <div class="p-2 text-xs text-gray-700">
            <div class="font-mono text-[8px] text-gray-500 truncate">{{ item.id }}</div>
            <div class="truncate text-[11px] font-medium">{{ item.payload?.label || '-' }}</div>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Loading indicators -->
    <div *ngIf="loading" class="text-center py-8">
      <div class="text-sm text-gray-500">Loading embeddings...</div>
    </div>
    
    <div *ngIf="loadingMore" class="text-center py-4">
      <div class="text-sm text-gray-500">Loading more...</div>
    </div>
    
    <!-- Infinite scroll trigger element -->
    <div id="infinite-scroll-trigger" class="h-10"></div>
    
    <div *ngIf="hasMoreData && !loadingMore && !loading" class="text-center py-4">
      <button (click)="loadMore()" 
              class="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 text-sm">
        Load More ({{ items.length }} loaded)
      </button>
    </div>
    
    <div *ngIf="!hasMoreData && !loading && items.length > 0" class="text-center py-4">
      <div class="text-sm text-gray-500">All embeddings loaded ({{ items.length }} total)</div>
    </div>
  </div>
  
  <!-- Edit Label Modal -->
  <div *ngIf="showEditLabelModal" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
    <div class="bg-white rounded-lg p-6 w-full max-w-2xl mx-4 max-h-[80vh] overflow-y-auto">
      <h3 class="text-lg font-semibold text-gray-900 mb-4">Edit Label</h3>
      
      <!-- Selected items preview -->
      <div class="mb-4">
        <div class="text-sm text-gray-600 mb-2">{{ editingItems.length }} item(s) selected:</div>
        <div class="grid grid-cols-4 md:grid-cols-6 gap-2 max-h-32 overflow-y-auto">
          <div *ngFor="let item of editingItems" class="border rounded overflow-hidden">
            <img [src]="getImageUrl(item)" alt="preview" class="w-full aspect-square object-contain bg-black" />
            <div class="p-1 text-xs text-center truncate">{{ item.payload?.label || '-' }}</div>
          </div>
        </div>
      </div>
      
      <!-- Label input -->
      <div class="mb-4">
        <label class="block text-sm font-medium text-gray-700 mb-2">New Label</label>
        <input 
          type="text" 
          [(ngModel)]="editLabelText" 
          placeholder="Enter new label for selected items"
          class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          (keyup.enter)="saveLabel()"
        />
      </div>
      
      <!-- Modal actions -->
      <div class="flex justify-end space-x-3">
        <button 
          (click)="closeEditLabelModal()" 
          [disabled]="savingLabel"
          class="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-md disabled:opacity-50 disabled:cursor-not-allowed">
          Cancel
        </button>
        <button 
          (click)="saveLabel()" 
          [disabled]="!editLabelText.trim() || savingLabel"
          class="px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-md disabled:opacity-50 disabled:cursor-not-allowed flex items-center">
          <svg *ngIf="savingLabel" class="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          {{ savingLabel ? 'Saving...' : 'Save Label' }}
        </button>
      </div>
    </div>
  </div>
  `,
})
export class ResultDetailComponent implements OnInit, OnDestroy, AfterViewInit {
  private readonly resultService = inject(ResultService);
  private readonly route = inject(ActivatedRoute);
  private readonly toastr = inject(ToastrService);
  runId: string = '';
  loading = false;
  loadingMore = false;
  items: Array<{ id: string; payload: any; imageUrl: string }> = [];
  groups: Record<string, Array<{ id: string; payload: any; imageUrl: string }>> = {};
  get groupsKeys(): string[] { return Object.keys(this.groups); }
  
  // Pagination
  private currentOffset = 0;
  private readonly pageSize = 200;
  hasMoreData = true;
  private observer: IntersectionObserver | null = null;
  
  // Prefetching cache
  private prefetchCache = new Map<number, Array<{ id: string; payload: any; imageUrl: string }>>();
  private prefetching = false;
  
  // Multiselect functionality
  selectedItems = new Set<string>();
  showEditLabelModal = false;
  editLabelText = '';
  editingItems: Array<{ id: string; payload: any; imageUrl: string }> = [];
  savingLabel = false;
  committingToThings = false;

  ngOnInit(): void {
    this.runId = String(this.route.snapshot.paramMap.get('id') || '');
    this.fetch();
  }

  ngAfterViewInit(): void {
    this.setupInfiniteScroll();
  }

  private setupInfiniteScroll(): void {
    // Use Intersection Observer for reliable infinite scroll
    this.observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting && !this.loadingMore && this.hasMoreData) {
          console.log('[DEBUG] Intersection Observer triggered - loading more data');
          this.loadMore();
        }
      });
    }, {
      root: null,
      rootMargin: '100px', // Trigger 100px before the element comes into view
      threshold: 0.1
    });

    // Start observing after a short delay to ensure DOM is ready
    setTimeout(() => {
      const triggerElement = document.getElementById('infinite-scroll-trigger');
      if (triggerElement && this.observer) {
        this.observer.observe(triggerElement);
        console.log('[DEBUG] Intersection Observer set up');
      }
    }, 1000);
  }

  ngOnDestroy(): void {
    if (this.observer) {
      this.observer.disconnect();
    }
    // Clear prefetch cache
    this.prefetchCache.clear();
  }

  private fetch(): void {
    this.loading = true;
    this.currentOffset = 0;
    this.hasMoreData = true;
    this.prefetchCache.clear(); // Clear cache on new fetch
    this.loadMore();
  }


  loadMore(): void {
    if (this.loadingMore || !this.hasMoreData) return;
    
    console.log('[DEBUG] Loading more data, offset:', this.currentOffset);
    this.loadingMore = true;
    
    // Check if we have prefetched data for this offset
    const cachedData = this.prefetchCache.get(this.currentOffset);
    if (cachedData) {
      console.log('[DEBUG] Using prefetched data for offset:', this.currentOffset);
      this.addItemsToDisplay(cachedData);
      this.prefetchCache.delete(this.currentOffset);
      this.loadingMore = false;
      
      // Trigger prefetch for next page
      this.prefetchNextPage();
      return;
    }
    
    // Load from API
    this.resultService.getEmbeddings(this.runId, this.pageSize, this.currentOffset).subscribe({
      next: (res) => {
        const newItems = (res.items || []).map(it => ({
          id: it.id,
          payload: it.payload,
          imageUrl: this.resultService.getImageUrl(this.runId, it.id),
        }));
        
        this.addItemsToDisplay(newItems);
        this.loadingMore = false;
        
        // Trigger prefetch for next page
        this.prefetchNextPage();
        
        console.log('[DEBUG] Loaded', newItems.length, 'items, total:', this.items.length, 'hasMore:', this.hasMoreData);
      },
      error: (err) => {
        console.error('Error fetching embeddings:', err);
        this.loadingMore = false;
      },
      complete: () => { 
        this.loading = false;
      },
    });
  }

  private addItemsToDisplay(newItems: Array<{ id: string; payload: any; imageUrl: string }>): void {
    if (this.currentOffset === 0) {
      // First load - replace items
      this.items = newItems;
    } else {
      // Subsequent loads - append items
      this.items = [...this.items, ...newItems];
    }
    
    this.groupByFrame();
    this.currentOffset += this.pageSize;
    this.hasMoreData = newItems.length === this.pageSize;
  }

  private prefetchNextPage(): void {
    if (this.prefetching || !this.hasMoreData) return;
    
    const nextOffset = this.currentOffset;
    if (this.prefetchCache.has(nextOffset)) {
      console.log('[DEBUG] Next page already prefetched for offset:', nextOffset);
      return;
    }
    
    console.log('[DEBUG] Prefetching next page, offset:', nextOffset);
    this.prefetching = true;
    
    this.resultService.getEmbeddings(this.runId, this.pageSize, nextOffset).subscribe({
      next: (res) => {
        const prefetchedItems = (res.items || []).map(it => ({
          id: it.id,
          payload: it.payload,
          imageUrl: this.resultService.getImageUrl(this.runId, it.id),
        }));
        
        // Cache the prefetched data
        this.prefetchCache.set(nextOffset, prefetchedItems);
        this.prefetching = false;
        
        console.log('[DEBUG] Prefetched', prefetchedItems.length, 'items for offset:', nextOffset);
      },
      error: (err) => {
        console.error('Error prefetching embeddings:', err);
        this.prefetching = false;
      }
    });
  }

  private groupByFrame(): void {
    // Sort items by frame_index first
    this.items.sort((a, b) => (a.payload?.frame_index || 0) - (b.payload?.frame_index || 0));
    
    // Group by frame for display
    const out: Record<string, Array<{ id: string; payload: any; imageUrl: string }>> = {};
    for (const it of this.items) {
      const frameIndex = it.payload?.frame_index || 0;
      const key = `Frame ${frameIndex}`;
      (out[key] = out[key] || []).push(it);
    }
    this.groups = out;
  }

  // Multiselect methods
  toggleSelection(itemId: string): void {
    if (this.selectedItems.has(itemId)) {
      this.selectedItems.delete(itemId);
    } else {
      this.selectedItems.add(itemId);
    }
  }

  clearSelection(): void {
    this.selectedItems.clear();
  }

  openEditLabelModal(): void {
    if (this.selectedItems.size === 0) return;
    
    // Get the selected items
    this.editingItems = this.items.filter(item => this.selectedItems.has(item.id));
    
    // Set initial label text (use the most common label if all items have the same label)
    const labels = this.editingItems.map(item => item.payload?.label).filter(Boolean);
    if (labels.length > 0 && labels.every(label => label === labels[0])) {
      this.editLabelText = labels[0];
    } else {
      this.editLabelText = '';
    }
    
    this.showEditLabelModal = true;
  }

  closeEditLabelModal(): void {
    this.showEditLabelModal = false;
    this.editLabelText = '';
    this.editingItems = [];
    this.savingLabel = false; // Reset loading state when modal is closed
  }

  getImageUrl(item: any): string {
    const imageId = item.payload?.image_id;
    return this.resultService.getImageUrl(this.runId, item.id, imageId);
  }

  saveLabel(): void {
    if (!this.editLabelText.trim() || this.selectedItems.size === 0 || this.savingLabel) return;
    
    const newLabel = this.editLabelText.trim();
    const selectedIds = Array.from(this.selectedItems);
    
    // Set loading state
    this.savingLabel = true;
    
    console.log(`[DEBUG] Updating labels for ${selectedIds.length} items:`, selectedIds);
    console.log(`[DEBUG] New label: "${newLabel}"`);
    
    // Update labels in Milvus
    this.resultService.updateEmbeddingLabels(this.runId, selectedIds, newLabel).subscribe({
      next: (response) => {
        console.log(`[DEBUG] Label update response:`, response);
        
        // Update local data directly instead of refreshing
        this.updateLocalData(selectedIds, newLabel);
        
        // Clear selection and close modal
        this.clearSelection();
        this.closeEditLabelModal();
        
        console.log(`[DEBUG] Updated ${response.updated} items successfully`);
      },
      error: (err) => {
        console.error('Error updating labels:', err);
        // TODO: Show error message to user
      },
      complete: () => {
        // Always reset loading state when request completes (success or error)
        this.savingLabel = false;
      }
    });
  }

  private updateLocalData(selectedIds: string[], newLabel: string): void {
    // Update the items array directly
    this.items.forEach(item => {
      if (selectedIds.includes(item.id)) {
        if (item.payload) {
          item.payload.label = newLabel;
        }
      }
    });
    
    // Regroup items to reflect changes
    this.groupByFrame();
    
    console.log(`[DEBUG] Updated local data for ${selectedIds.length} items with label: "${newLabel}"`);
  }

  commitToMyThings(): void {
    if (this.selectedItems.size === 0 || this.committingToThings) return;
    
    const selectedIds = Array.from(this.selectedItems);
    this.committingToThings = true;
    
    console.log(`[DEBUG] Committing ${selectedIds.length} items to My Things collection`);
    
    this.resultService.commitToMyThings(this.runId, selectedIds).subscribe({
      next: (response) => {
        console.log(`[DEBUG] Commit response:`, response);
        
        // Update local data to mark items as committed
        this.updateLocalDataCommitted(selectedIds);
        
        // Show success toast
        this.toastr.success(`Successfully committed ${response.committed} items to My Things library`);
        
        // Clear selection
        this.clearSelection();
        
        console.log(`[DEBUG] Committed ${response.committed} items to My Things successfully`);
      },
      error: (err) => {
        console.error('Error committing to My Things:', err);
        this.toastr.error('Failed to commit items to My Things library');
      },
      complete: () => {
        this.committingToThings = false;
      }
    });
  }

  private updateLocalDataCommitted(selectedIds: string[]): void {
    // Update the items array to mark as committed
    this.items.forEach(item => {
      if (selectedIds.includes(item.id)) {
        if (item.payload) {
          item.payload.committed = true;
        }
      }
    });
    
    // Regroup items to reflect changes
    this.groupByFrame();
    
    console.log(`[DEBUG] Updated local data for ${selectedIds.length} items as committed`);
  }

}


