import { Component, OnInit, OnDestroy, AfterViewInit, inject } from '@angular/core';
import { CommonModule, NgFor, NgIf } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ResultService, EmbeddingItem } from '../../core/services/result.service';
import { ToastrService } from 'ngx-toastr';

@Component({
  selector: 'app-things',
  standalone: true,
  imports: [CommonModule, NgIf, NgFor, FormsModule],
  templateUrl: './things.component.html',
})
export class ThingsComponent implements OnInit, OnDestroy, AfterViewInit {
  private readonly resultService = inject(ResultService);
  private readonly toastr = inject(ToastrService);
  
  loading = false;
  loadingMore = false;
  items: Array<{ id: string; payload: any; imageUrl: string }> = [];
  
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

  ngOnInit(): void {
    this.fetch();
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
    return this.resultService.getThingsImageUrl(item);
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
        this.currentOffset = 0;
        this.hasMoreData = false;
        this.prefetchCache.clear();
        
        // Close modal
        this.closeDeleteAllModal();
        
        // Show success message (you could add a toast notification here)
        alert(`Successfully deleted all things!\n\nDeleted ${response.deleted_embeddings} embeddings and ${response.deleted_images} images.`);
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
