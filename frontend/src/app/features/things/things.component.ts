import { Component, OnInit, OnDestroy, AfterViewInit, inject } from '@angular/core';
import { CommonModule, NgFor, NgIf } from '@angular/common';
import { Router, RouterLink } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { ResultService, EmbeddingItem, ThingGroup, ThingGroupDetail } from '../../core/services/result.service';
import { ThingCardComponent } from '../../shared/components/thing-card.component';
import { ToastrService } from 'ngx-toastr';

@Component({
  selector: 'app-things',
  standalone: true,
  imports: [CommonModule, NgIf, NgFor, FormsModule, RouterLink, ThingCardComponent],
  templateUrl: './things.component.html',
})
export class ThingsComponent implements OnInit, OnDestroy, AfterViewInit {
  private readonly resultService = inject(ResultService);
  private readonly toastr = inject(ToastrService);
  private readonly router = inject(Router);
  // testing service removed
  
  loading = false;
  loadingMore = false;
  items: Array<{ id: string; payload: any; imageUrl: string }> = [];
  groups: ThingGroup[] = [];
  
  // Search functionality
  searchQuery = '';
  isSearching = false;
  searchResults: Array<{ id: string; payload: any; imageUrl: string }> = [];
  private searchOffset = 0;
  private searchHasMoreData = true;
  
  // Pagination
  private currentOffset = 0;
  private readonly pageSize = 200;
  hasMoreData = true;
  private observer: IntersectionObserver | null = null;
  
  // Prefetching cache
  private prefetchCache = new Map<number, Array<{ id: string; payload: any; imageUrl: string }>>();
  private prefetching = false;
  
  // Delete all functionality
  showDeleteAllModal = false;
  isDeletingAll = false;
  
  // Multiselect functionality
  selectedItems = new Set<string>();
  showEditLabelModal = false;
  editLabelText = '';
  editingItems: Array<{ id: string; payload: any; imageUrl: string }> = [];
  savingLabel = false;
  deletingSelected = false;

  // Grouping UI state
  creatingGroup = false;
  showGroupModal = false;
  groupLoading = false;
  activeGroup: ThingGroupDetail | null = null;
  // Track one selected group (id) for add-to-group action
  selectedGroupId: string | null = null;
  
  // Delete group functionality
  deletingGroup = false;
  groupToDelete: ThingGroup | null = null;

  // Edit crop state
  showEditCrop = false;
  editEmbeddingId: string | null = null;
  editFullImageUrl = '';
  editBBox: { x1: number; y1: number; x2: number; y2: number } = { x1: 0, y1: 0, x2: 0, y2: 0 };
  editFullNaturalW = 0;
  editFullNaturalH = 0;
  editFullRenderW = 0;
  editFullRenderH = 0;
  // Drag state for bbox
  isDraggingBBox = false;
  dragStartRenderX = 0;
  dragStartRenderY = 0;
  dragStartBBox: { x1: number; y1: number; x2: number; y2: number } = { x1: 0, y1: 0, x2: 0, y2: 0 };
  dragContainerRect: DOMRect | null = null;
  // Resize state for bbox
  isResizingBBox = false;
  resizeHandle: 'n' | 's' | 'e' | 'w' | 'ne' | 'nw' | 'se' | 'sw' | null = null;
  // Saving crop state
  savingCrop = false;

  // Negatives tab state
  activeTab: 'things' | 'negatives' = 'things';
  negatives: string[] = [];
  loadingNegatives = false;

  ngOnInit(): void {
    this.fetch();
    this.loadGroups();
  }

  ngAfterViewInit(): void {
    // Set up intersection observer for infinite scroll
    this.observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting && this.hasMoreData && !this.loadingMore) {
          console.log('[DEBUG] Infinite scroll triggered');
          this.loadMore();
        }
      });
    }, {
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

  switchTab(tab: 'things' | 'negatives'): void {
    this.activeTab = tab;
    if (tab === 'negatives' && !this.loadingNegatives) {
      this.loadNegatives();
    }
  }

  private loadNegatives(): void {
    this.loadingNegatives = true;
    this.resultService.listNegatives().subscribe({
      next: (res) => {
        this.negatives = Array.isArray(res.items) ? res.items : [];
      },
      error: (err) => {
        console.error('Failed to load negatives', err);
        this.toastr.error('Failed to load negatives');
      },
      complete: () => { this.loadingNegatives = false; }
    });
  }

  removeNegative(objectName: string): void {
    this.resultService.deleteNegative(objectName).subscribe({
      next: () => {
        this.negatives = this.negatives.filter(x => x !== objectName);
        this.toastr.success('Removed negative');
      },
      error: (err) => {
        console.error('Failed to remove negative', err);
        this.toastr.error('Failed to remove negative');
      }
    });
  }

  getNegativeImageUrl(objectName: string): string {
    return this.resultService.getNegativeImageUrl(objectName);
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

  private loadGroups(): void {
    this.resultService.getThingGroups().subscribe({
      next: (res) => {
        this.groups = res || [];
      },
      error: (err) => {
        console.error('Failed to load thing groups', err);
      }
    });
  }

  // Removed ungrouped filter; backend already returns ungrouped via group_id == "unknown"

  // Search functionality
  onSearchInput(event: Event): void {
    const target = event.target as HTMLInputElement;
    this.searchQuery = target.value.trim();
    
    if (this.searchQuery.length === 0) {
      // Clear search and show all items
      this.clearSearch();
    } else if (this.searchQuery.length >= 2) {
      // Perform search with debounce
      this.debounceSearch();
    }
  }

  private debounceTimer: any;
  private debounceSearch(): void {
    clearTimeout(this.debounceTimer);
    this.debounceTimer = setTimeout(() => {
      this.performSearch();
    }, 300); // 300ms debounce
  }

  private performSearch(): void {
    if (this.searchQuery.length < 2) return;
    
    this.isSearching = true;
    this.searchOffset = 0;
    this.searchHasMoreData = true;
    this.searchResults = [];
    
    this.resultService.searchThingsEmbeddings(this.searchQuery, this.pageSize, this.searchOffset).subscribe({
      next: (res) => {
        const newItems = (res.items || []).map(it => ({
          id: it.id,
          payload: it.payload,
          imageUrl: this.resultService.getThingsImageUrl(it),
        }));
        
        this.searchResults = newItems;
        this.searchOffset += this.pageSize;
        this.searchHasMoreData = newItems.length === this.pageSize;
        this.isSearching = false;
      },
      error: (err) => {
        console.error('Error searching things:', err);
        this.isSearching = false;
        this.toastr.error('Search failed');
      }
    });
  }

  clearSearch(): void {
    this.searchQuery = '';
    this.searchResults = [];
    this.isSearching = false;
    this.searchOffset = 0;
    this.searchHasMoreData = true;
  }

  get displayItems(): Array<{ id: string; payload: any; imageUrl: string }> {
    return this.searchQuery.length >= 2 ? this.searchResults : this.items;
  }

  get hasMoreDisplayData(): boolean {
    return this.searchQuery.length >= 2 ? this.searchHasMoreData : this.hasMoreData;
  }

  loadMore(): void {
    if (this.loadingMore || !this.hasMoreDisplayData) return;
    
    // Handle search results
    if (this.searchQuery.length >= 2) {
      this.loadMoreSearchResults();
      return;
    }
    
    console.log('[DEBUG] Loading more things, offset:', this.currentOffset);
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
    this.resultService.getThingsEmbeddings(this.pageSize, this.currentOffset).subscribe({
      next: (res) => {
        const newItems = (res.items || []).map(it => ({
          id: it.id,
          payload: it.payload,
          imageUrl: this.resultService.getThingsImageUrl(it),
        }));
        
        this.addItemsToDisplay(newItems);
        this.loadingMore = false;
        
        // Trigger prefetch for next page
        this.prefetchNextPage();
        
        console.log('[DEBUG] Loaded', newItems.length, 'things, total:', this.items.length, 'hasMore:', this.hasMoreData);
      },
      error: (err) => {
        console.error('Error fetching things:', err);
        this.loadingMore = false;
      },
      complete: () => { 
        this.loading = false;
      },
    });
  }

  private loadMoreSearchResults(): void {
    if (this.loadingMore || !this.searchHasMoreData) return;
    
    console.log('[DEBUG] Loading more search results, offset:', this.searchOffset);
    this.loadingMore = true;
    
    this.resultService.searchThingsEmbeddings(this.searchQuery, this.pageSize, this.searchOffset).subscribe({
      next: (res) => {
        const newItems = (res.items || []).map(it => ({
          id: it.id,
          payload: it.payload,
          imageUrl: this.resultService.getThingsImageUrl(it),
        }));
        
        this.searchResults = [...this.searchResults, ...newItems];
        this.searchOffset += this.pageSize;
        this.searchHasMoreData = newItems.length === this.pageSize;
        this.loadingMore = false;
        
        console.log('[DEBUG] Loaded', newItems.length, 'search results, total:', this.searchResults.length, 'hasMore:', this.searchHasMoreData);
      },
      error: (err) => {
        console.error('Error fetching search results:', err);
        this.loadingMore = false;
      }
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
    
    this.resultService.getThingsEmbeddings(this.pageSize, nextOffset).subscribe({
      next: (res) => {
        const prefetchedItems = (res.items || []).map(it => ({
          id: it.id,
          payload: it.payload,
          imageUrl: this.resultService.getThingsImageUrl(it),
        }));
        
        // Cache the prefetched data
        this.prefetchCache.set(nextOffset, prefetchedItems);
        this.prefetching = false;
        
        console.log('[DEBUG] Prefetched', prefetchedItems.length, 'things for offset:', nextOffset);
      },
      error: (err) => {
        console.error('Error prefetching things:', err);
        this.prefetching = false;
      }
    });
  }

  getImageUrl(item: any): string {
    // Prefer locally cached URL (may include cache-busting query) if present
    return item?.imageUrl || this.resultService.getThingsImageUrl(item);
  }

  // Delete all functionality
  openDeleteAllModal(): void {
    this.showDeleteAllModal = true;
    this.isDeletingAll = false;
  }

  closeDeleteAllModal(): void {
    this.showDeleteAllModal = false;
    this.isDeletingAll = false;
  }

  confirmDeleteAll(): void {
    this.isDeletingAll = true;
    
    this.resultService.deleteAllThings().subscribe({
      next: (response) => {
        console.log('Delete all things successful:', response);
        
        // Clear all local data
        this.items = [];
        this.groups = [];
        this.currentOffset = 0;
        this.hasMoreData = false;
        this.prefetchCache.clear();
        
        // Close modal
        this.closeDeleteAllModal();
        
        // Show success message (you could add a toast notification here)
        alert(`Successfully deleted all things!\n\nDeleted ${response.deleted_embeddings} embeddings, ${response.deleted_images} images, and all groups.`);
      },
      error: (err) => {
        console.error('Failed to delete all things:', err);
        this.isDeletingAll = false;
        
        // Show error message
        alert(`Failed to delete all things: ${err.error?.detail || err.message || 'Unknown error'}`);
      }
    });
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
    this.selectedGroupId = null;
  }

  groupSelected(): void {
    if (this.creatingGroup) return;
    const selectedIds = Array.from(this.selectedItems);
    if (selectedIds.length === 0) return;

    // If a group is selected, add to that group; else create a new group (need at least 2 items)
    if (this.selectedGroupId) {
      this.creatingGroup = true;
      this.resultService.addThingsToGroup(this.selectedGroupId, selectedIds).subscribe({
        next: (res) => {
          // Remove newly grouped items from local lists
          this.removeItemsByIds(selectedIds);
          // Optionally refresh groups list or increment count
          const g = this.groups.find(x => x.id === this.selectedGroupId);
          if (g) {
            g.count = Math.max(0, (g.count || 0) + selectedIds.length);
          }
          this.clearSelection();
          this.toastr.success('Added to group');
        },
        error: (err) => {
          console.error('Failed to add to group', err);
          this.toastr.error('Failed to add to group');
        },
        complete: () => {
          this.creatingGroup = false;
        }
      });
      return;
    }

    if (selectedIds.length < 2) return;
    this.creatingGroup = true;
    this.resultService.createThingGroup(selectedIds).subscribe({
      next: (group) => {
        // Apply label locally to selected items to reflect grouping label
        const newLabel = group.name;
        this.updateLocalDataLabels(selectedIds, newLabel);
        // Add group to list and clear selection
        this.groups = [group, ...this.groups];
        // Remove grouped items from local lists immediately
        this.removeItemsByIds(selectedIds);
        this.clearSelection();
      },
      error: (err) => {
        console.error('Failed to create thing group', err);
        this.toastr.error('Failed to create group');
      },
      complete: () => {
        this.creatingGroup = false;
      }
    });
  }

  // Toggle selecting a group (only one allowed)
  toggleGroupSelection(groupId: string): void {
    if (this.selectedGroupId === groupId) {
      this.selectedGroupId = null;
    } else {
      this.selectedGroupId = groupId;
    }
  }

  openGroup(groupId: string): void {
    // Navigate to group page
    this.router.navigate(['/things/groups', groupId]);
  }

  closeGroupModal(): void {
    this.showGroupModal = false;
    this.activeGroup = null;
  }

  // Delete group functionality
  openDeleteGroupModal(group: ThingGroup): void {
    this.groupToDelete = group;
    this.deletingGroup = false;
  }

  closeDeleteGroupModal(): void {
    this.groupToDelete = null;
    this.deletingGroup = false;
  }

  confirmDeleteGroup(): void {
    if (!this.groupToDelete || this.deletingGroup) return;
    
    this.deletingGroup = true;
    
    this.resultService.deleteThingGroup(this.groupToDelete.id).subscribe({
      next: (response) => {
        // Remove group from local list
        this.groups = this.groups.filter(g => g.id !== this.groupToDelete!.id);
        
        this.toastr.success(`Group "${this.groupToDelete!.name}" deleted. ${response.ungrouped_items} items moved to ungrouped.`);
        this.closeDeleteGroupModal();
      },
      error: (err) => {
        console.error('Failed to delete group:', err);
        this.toastr.error('Failed to delete group');
        this.deletingGroup = false;
      }
    });
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

  openEditCropModal(): void {
    if (this.selectedItems.size !== 1) return;
    const id = Array.from(this.selectedItems)[0];
    const item = this.items.find(it => it.id === id) || this.searchResults.find(it => it.id === id);
    if (!item) return;
    const imageId = item.payload?.image_id;
    if (!imageId) {
      this.toastr.error('Missing image_id for crop');
      return;
    }
    // Preload bbox if present
    const bbox = item.payload?.bbox;
    if (Array.isArray(bbox) && bbox.length === 4) {
      this.editBBox = { x1: Math.round(bbox[0]), y1: Math.round(bbox[1]), x2: Math.round(bbox[2]), y2: Math.round(bbox[3]) };
    } else {
      this.editBBox = { x1: 0, y1: 0, x2: 0, y2: 0 };
    }
    this.editEmbeddingId = id;
    this.editFullImageUrl = `/api/train/things/image/${encodeURIComponent(imageId)}_full.jpg`;
    this.showEditCrop = true;
  }

  closeEditCropModal(): void {
    this.showEditCrop = false;
    this.editEmbeddingId = null;
    this.editFullImageUrl = '';
    this.editFullNaturalW = 0;
    this.editFullNaturalH = 0;
    this.editFullRenderW = 0;
    this.editFullRenderH = 0;
  }

  saveEditCrop(): void {
    if (!this.editEmbeddingId) return;
    if (this.savingCrop) return;
    this.savingCrop = true;
    const x1 = Math.round(this.editBBox.x1);
    const y1 = Math.round(this.editBBox.y1);
    const x2 = Math.round(this.editBBox.x2);
    const y2 = Math.round(this.editBBox.y2);
    this.resultService.editThingCrop(this.editEmbeddingId, { x1, y1, x2, y2 }).subscribe({
      next: (res) => {
        // Update local payload bbox and refresh image url (bust cache)
        const upd = (arr: Array<{ id: string; payload: any; imageUrl: string }>) => {
          return arr.map(it => it.id === this.editEmbeddingId ? {
            ...it,
            payload: { ...it.payload, bbox: [x1, y1, x2, y2] },
            imageUrl: `${this.resultService.getThingsImageUrl(it)}?t=${Date.now()}`
          } : it);
        };
        this.items = upd(this.items);
        this.searchResults = upd(this.searchResults);
        this.toastr.success('Crop updated');
        this.closeEditCropModal();
      },
      error: (err) => {
        console.error('Failed to update crop', err);
        this.toastr.error(err?.error?.detail || 'Failed to update crop');
      },
      complete: () => {
        this.savingCrop = false;
      }
    });
  }

  onFullImageLoad(event: Event): void {
    const img = event.target as HTMLImageElement;
    this.editFullNaturalW = Number(img.naturalWidth || 0);
    this.editFullNaturalH = Number(img.naturalHeight || 0);
    this.editFullRenderW = Number(img.clientWidth || img.width || 0);
    this.editFullRenderH = Number(img.clientHeight || img.height || 0);
  }

  get hasEditBBox(): boolean {
    const { x1, y1, x2, y2 } = this.editBBox;
    return isFinite(x1) && isFinite(y1) && isFinite(x2) && isFinite(y2) && x2 > x1 && y2 > y1;
  }

  getBboxOverlayStyle(): { [key: string]: string } {
    if (!this.hasEditBBox || this.editFullNaturalW <= 0 || this.editFullNaturalH <= 0 || this.editFullRenderW <= 0 || this.editFullRenderH <= 0) {
      return {};
    }
    const scaleX = this.editFullRenderW / this.editFullNaturalW;
    const scaleY = this.editFullRenderH / this.editFullNaturalH;
    const left = Math.max(0, Math.round(this.editBBox.x1 * scaleX));
    const top = Math.max(0, Math.round(this.editBBox.y1 * scaleY));
    const width = Math.max(1, Math.round((this.editBBox.x2 - this.editBBox.x1) * scaleX));
    const height = Math.max(1, Math.round((this.editBBox.y2 - this.editBBox.y1) * scaleY));
    return {
      left: `${left}px`,
      top: `${top}px`,
      width: `${width}px`,
      height: `${height}px`,
      border: '2px solid #22c55e', // tailwind green-500
      boxSizing: 'border-box',
    };
  }

  private clamp(val: number, min: number, max: number): number {
    return Math.max(min, Math.min(max, val));
  }

  startBboxDrag(event: MouseEvent): void {
    if (!this.hasEditBBox) return;
    event.preventDefault();
    const target = event.currentTarget as HTMLElement | null;
    const parent = (target && target.parentElement) ? target.parentElement : null;
    const rect = parent ? parent.getBoundingClientRect() : null;
    if (!rect) return;
    this.dragContainerRect = rect;
    this.dragStartRenderX = event.clientX - rect.left;
    this.dragStartRenderY = event.clientY - rect.top;
    this.dragStartBBox = { ...this.editBBox };
    this.isDraggingBBox = true;
  }

  onBboxPointerMove(event: MouseEvent): void {
    if (!this.dragContainerRect) return;
    event.preventDefault();
    const renderX = event.clientX - this.dragContainerRect.left;
    const renderY = event.clientY - this.dragContainerRect.top;
    const dRenderX = renderX - this.dragStartRenderX;
    const dRenderY = renderY - this.dragStartRenderY;
    const scaleX = this.editFullRenderW > 0 ? (this.editFullRenderW / this.editFullNaturalW) : 1;
    const scaleY = this.editFullRenderH > 0 ? (this.editFullRenderH / this.editFullNaturalH) : 1;
    const dNatX = scaleX ? dRenderX / scaleX : 0;
    const dNatY = scaleY ? dRenderY / scaleY : 0;
    const minW = 1;
    const minH = 1;

    if (this.isDraggingBBox) {
      const wNat = this.dragStartBBox.x2 - this.dragStartBBox.x1;
      const hNat = this.dragStartBBox.y2 - this.dragStartBBox.y1;
      let nx1 = this.dragStartBBox.x1 + dNatX;
      let ny1 = this.dragStartBBox.y1 + dNatY;
      nx1 = this.clamp(nx1, 0, Math.max(0, this.editFullNaturalW - wNat));
      ny1 = this.clamp(ny1, 0, Math.max(0, this.editFullNaturalH - hNat));
      const nx2 = nx1 + wNat;
      const ny2 = ny1 + hNat;
      this.editBBox = { x1: Math.round(nx1), y1: Math.round(ny1), x2: Math.round(nx2), y2: Math.round(ny2) };
      return;
    }

    if (this.isResizingBBox && this.resizeHandle) {
      let { x1, y1, x2, y2 } = this.dragStartBBox;
      // Adjust based on handle
      if (this.resizeHandle.includes('e')) {
        x2 = this.clamp(x2 + dNatX, x1 + minW, this.editFullNaturalW);
      }
      if (this.resizeHandle.includes('w')) {
        x1 = this.clamp(x1 + dNatX, 0, x2 - minW);
      }
      if (this.resizeHandle.includes('s')) {
        y2 = this.clamp(y2 + dNatY, y1 + minH, this.editFullNaturalH);
      }
      if (this.resizeHandle.includes('n')) {
        y1 = this.clamp(y1 + dNatY, 0, y2 - minH);
      }
      this.editBBox = { x1: Math.round(x1), y1: Math.round(y1), x2: Math.round(x2), y2: Math.round(y2) };
      return;
    }
  }

  endBboxDrag(): void {
    this.isDraggingBBox = false;
    this.isResizingBBox = false;
    this.resizeHandle = null;
    this.dragContainerRect = null;
  }

  startBboxResize(handle: 'n' | 's' | 'e' | 'w' | 'ne' | 'nw' | 'se' | 'sw', event: MouseEvent): void {
    event.preventDefault();
    event.stopPropagation();
    const parent = (event.currentTarget as HTMLElement)?.parentElement;
    const container = parent?.parentElement || null;
    const rect = container ? container.getBoundingClientRect() : null;
    if (!rect) return;
    this.dragContainerRect = rect;
    this.dragStartRenderX = event.clientX - rect.left;
    this.dragStartRenderY = event.clientY - rect.top;
    this.dragStartBBox = { ...this.editBBox };
    this.isDraggingBBox = false;
    this.isResizingBBox = true;
    this.resizeHandle = handle;
  }

  closeEditLabelModal(): void {
    this.showEditLabelModal = false;
    this.editLabelText = '';
    this.editingItems = [];
  }

  saveLabel(): void {
    if (!this.editLabelText.trim() || this.selectedItems.size === 0 || this.savingLabel) return;
    
    const newLabel = this.editLabelText.trim();
    const selectedIds = Array.from(this.selectedItems);
    
    // Set loading state
    this.savingLabel = true;
    
    console.log(`[DEBUG] Updating things labels for ${selectedIds.length} items:`, selectedIds);
    console.log(`[DEBUG] New label: "${newLabel}"`);
    
    this.resultService.updateThingsLabels(selectedIds, newLabel).subscribe({
      next: (response) => {
        console.log(`[DEBUG] Update response:`, response);
        
        // Update local data
        this.updateLocalDataLabels(selectedIds, newLabel);
        
        // Show success toast
        this.toastr.success(`Successfully updated ${response.updated} items with label: "${newLabel}"`);
        
        // Clear selection and close modal
        this.clearSelection();
        this.closeEditLabelModal();
        
        console.log(`[DEBUG] Updated ${response.updated} items successfully`);
      },
      error: (err) => {
        console.error('Error updating things labels:', err);
        this.toastr.error('Failed to update labels');
      },
      complete: () => {
        // Always reset loading state when request completes (success or error)
        this.savingLabel = false;
      }
    });
  }

  private updateLocalDataLabels(selectedIds: string[], newLabel: string): void {
    // Update the local items array to reflect the new labels
    this.items = this.items.map(item => {
      if (selectedIds.includes(item.id)) {
        return {
          ...item,
          payload: {
            ...item.payload,
            label: newLabel
          }
        };
      }
      return item;
    });
  }

  private removeItemsByIds(ids: string[]): void {
    const idSet = new Set(ids.map(String));
    this.items = this.items.filter(it => !idSet.has(String(it.id)));
    this.searchResults = this.searchResults.filter(it => !idSet.has(String(it.id)));
  }

  deleteSelected(): void {
    if (this.selectedItems.size === 0 || this.deletingSelected) return;
    
    const selectedIds = Array.from(this.selectedItems);
    this.deletingSelected = true;
    
    console.log(`[DEBUG] Deleting ${selectedIds.length} selected things:`, selectedIds);
    
    this.resultService.deleteSelectedThings(selectedIds).subscribe({
      next: (response) => {
        console.log(`[DEBUG] Delete response:`, response);
        
        // Remove deleted items from local data
        this.removeDeletedItemsFromLocalData(selectedIds);
        
        // Show success toast
        this.toastr.success(`Successfully deleted ${response.deleted} items from My Things library`);
        
        // Clear selection
        this.clearSelection();
        
        console.log(`[DEBUG] Deleted ${response.deleted} items successfully`);
      },
      error: (err) => {
        console.error('Error deleting selected things:', err);
        this.toastr.error('Failed to delete selected items');
      },
      complete: () => {
        this.deletingSelected = false;
      }
    });
  }

  private removeDeletedItemsFromLocalData(selectedIds: string[]): void {
    // Remove deleted items from the local items array
    this.items = this.items.filter(item => !selectedIds.includes(item.id));
  }
}
