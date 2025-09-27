import { Component, OnInit, OnDestroy, ChangeDetectorRef, ChangeDetectionStrategy, TrackByFunction } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import { SystemService } from '../../core/services/system.service';
import { TrainingService } from '../../core/services/training.service';
import { SocketService } from '../../core/services/socket.service';
import { DeleteConfirmationModalComponent } from '../../shared/components/delete-confirmation-modal/delete-confirmation-modal.component';

@Component({
  selector: 'app-results',
  imports: [CommonModule, RouterLink, DeleteConfirmationModalComponent],
  templateUrl: './results.component.html',
  styleUrl: './results.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class ResultsComponent implements OnInit, OnDestroy {
  runs: Array<{ _id: string; status: string; file_name: string; started_at: number; completed_at: number | null; frames_processed?: number; total_frames?: number; elapsed_s?: number; _deleting?: boolean }> = [];
  total = 0;
  page = 1;
  pageSize = 20;
  loading = false;

  // Delete modal state
  showDeleteModal = false;
  deleteTargetId: string | null = null;
  isDeleting = false;

  private trainingId: string | null = null;
  private newJobsPollTimer: any = null;
  private lastUpdateTime = 0;
  private readonly UPDATE_THROTTLE_MS = 100; // Throttle updates to prevent rapid re-renders

  // TrackBy function to help Angular identify which rows have changed
  trackByRunId: TrackByFunction<{ _id: string; status: string; file_name: string; started_at: number; completed_at: number | null; frames_processed?: number; total_frames?: number; elapsed_s?: number; _deleting?: boolean }> = (index, run) => run._id;

  constructor(private readonly system: SystemService,
              private readonly socket: SocketService,
              private readonly training: TrainingService,
              private readonly cdr: ChangeDetectorRef) {}

  ngOnInit(): void {
    this.load();
    this.startNewJobsPolling();
    // Initial reconcile in case table shows running but backend is idle
    this.training.getTrainingStatus().subscribe({
      next: (st) => this.reconcileIfIdle(st),
      error: () => {}
    });
    // Subscribe to training status to live-update the current run
    this.socket.on<any>('training_status', (t) => {
      try { console.debug('[Results] training_status', t); } catch {}
      const id: string | undefined = t?.training_id;
      if (id && !this.trainingId) {
        // New run started; re-fetch first page and pin id
        this.trainingId = id;
        this.load(1);
      }
      if (id) {
        // Patch the row if present
        const idx = this.runs.findIndex(r => r._id === id);
        if (idx >= 0) {
          const status = t?.state ?? this.runs[idx].status;
          const started_at = typeof t?.started_at === 'number' ? t.started_at : this.runs[idx].started_at;
          const completed_at = typeof t?.ended_at === 'number' ? t.ended_at : this.runs[idx].completed_at;
          const frames_processed = typeof t?.frames_processed === 'number' ? t.frames_processed : this.runs[idx].frames_processed;
          const total_frames = typeof t?.total_frames === 'number' ? t.total_frames : this.runs[idx].total_frames;
          const elapsed_s = typeof t?.elapsed_s === 'number' ? t.elapsed_s : this.runs[idx].elapsed_s;
          
          // Only update if data actually changed to reduce unnecessary re-renders
          const currentRun = this.runs[idx];
          if (currentRun.status !== status || 
              currentRun.started_at !== started_at || 
              currentRun.completed_at !== completed_at || 
              currentRun.frames_processed !== frames_processed || 
              currentRun.total_frames !== total_frames || 
              currentRun.elapsed_s !== elapsed_s) {
            this.runs[idx] = { ...this.runs[idx], status, started_at, completed_at, frames_processed, total_frames, elapsed_s };
            this.throttledDetectChanges();
          }
        } else {
          // General fallback: update any visible running rows with progress from status message
          const runningRows = this.runs
            .map((r, i) => ({ r, i }))
            .filter(x => x.r.status === 'running');
          if (runningRows.length > 0) {
            let hasChanges = false;
            for (const x of runningRows) {
              const status = t?.state ?? x.r.status;
              const started_at = typeof t?.started_at === 'number' ? t.started_at : x.r.started_at;
              const completed_at = typeof t?.ended_at === 'number' ? t.ended_at : x.r.completed_at;
              const frames_processed = typeof t?.frames_processed === 'number' ? t.frames_processed : x.r.frames_processed;
              const total_frames = typeof t?.total_frames === 'number' ? t.total_frames : x.r.total_frames;
              const elapsed_s = typeof t?.elapsed_s === 'number' ? t.elapsed_s : x.r.elapsed_s;
              
              // Only update if data actually changed
              if (x.r.status !== status || 
                  x.r.started_at !== started_at || 
                  x.r.completed_at !== completed_at || 
                  x.r.frames_processed !== frames_processed || 
                  x.r.total_frames !== total_frames || 
                  x.r.elapsed_s !== elapsed_s) {
                this.runs[x.i] = { ...x.r, status, started_at, completed_at, frames_processed, total_frames, elapsed_s };
                hasChanges = true;
              }
            }
            if (hasChanges) {
              this.throttledDetectChanges();
            }
          }
        }
      } else {
        // Fallback: if no training_id is provided, and exactly one running row exists, update that row
        const runningIdx = this.runs.findIndex(r => r.status === 'running');
        if (runningIdx >= 0) {
          const frames_processed = typeof t?.frames_processed === 'number' ? t.frames_processed : this.runs[runningIdx].frames_processed;
          const total_frames = typeof t?.total_frames === 'number' ? t.total_frames : this.runs[runningIdx].total_frames;
          const elapsed_s = typeof t?.elapsed_s === 'number' ? t.elapsed_s : this.runs[runningIdx].elapsed_s;
          
          // Only update if data actually changed
          const currentRun = this.runs[runningIdx];
          if (currentRun.frames_processed !== frames_processed || 
              currentRun.total_frames !== total_frames || 
              currentRun.elapsed_s !== elapsed_s) {
            this.runs[runningIdx] = { ...this.runs[runningIdx], frames_processed, total_frames, elapsed_s };
            this.throttledDetectChanges();
          }
        }
      }
      const state = t?.state;
      if (state && state !== 'running') {
        this.trainingId = null;
        // If backend says idle but we show running rows, reconcile them to error
        this.reconcileIfIdle({ state });
      }
    });
  }

  load(page: number = this.page): void {
    this.loading = true;
    this.training.listTrainingRuns(page, this.pageSize).subscribe({
      next: (res) => {
        this.total = res.total || 0;
        this.runs = res.items || [];
        this.page = page;
        // Trigger change detection for OnPush strategy
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.loading = false;
        this.cdr.detectChanges();
      },
      complete: () => { 
        this.loading = false; 
        this.cdr.detectChanges();
      }
    });
  }

  totalPages(): number {
    return this.pageSize > 0 ? Math.max(1, Math.ceil(this.total / this.pageSize)) : 1;
  }

  nextPage(): void { if (this.page < this.totalPages()) { this.load(this.page + 1); } }
  prevPage(): void { if (this.page > 1) { this.load(this.page - 1); } }

  ngOnDestroy(): void {
    // No-op: socket lifecycles at app root
    this.stopNewJobsPolling();
  }

  fmtDuration(totalSeconds?: number | null): string {
    if (!totalSeconds || totalSeconds < 0) return '-';
    const s = Math.floor(totalSeconds);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    const pad = (n: number) => n.toString().padStart(2, '0');
    return h > 0 ? `${h}:${pad(m)}:${pad(sec)}` : `${m}:${pad(sec)}`;
  }

  // Only poll to discover new jobs (page 1), not for progress updates
  private startNewJobsPolling(): void {
    if (!this.newJobsPollTimer) {
      this.newJobsPollTimer = setInterval(() => {
        if (this.page !== 1) return;
        this.training.listTrainingRuns(1, this.pageSize).subscribe({
          next: (res) => {
            const items = res.items || [];
            // Merge new head items that are not in current runs
            const existingIds = new Set(this.runs.map(r => r._id));
            const newOnes = items.filter(i => !existingIds.has(i._id));
            if (newOnes.length > 0) {
              this.runs = [...newOnes, ...this.runs].slice(0, this.pageSize);
              this.total = res.total || this.total;
            }
          },
          error: () => {}
        });
        // Also reconcile if backend is idle but we show running rows
        this.training.getTrainingStatus().subscribe({
          next: (st) => this.reconcileIfIdle(st),
          error: () => {}
        });
      }, 7000);
    }
  }

  private stopNewJobsPolling(): void {
    if (this.newJobsPollTimer) {
      clearInterval(this.newJobsPollTimer);
      this.newJobsPollTimer = null;
    }
  }

  private reconcileIfIdle(status: any): void {
    const st = status?.state;
    if (st === 'idle') {
      const hasRunning = this.runs.some(r => r.status === 'running');
      if (hasRunning) {
        this.training.reconcileRunning().subscribe({
          next: () => { this.load(this.page); },
          error: () => { this.load(this.page); }
        });
      }
    }
  }

  deleteRun(id: string): void {
    if (!id) return;
    this.deleteTargetId = id;
    this.showDeleteModal = true;
    this.cdr.detectChanges();
  }

  onDeleteConfirm(): void {
    if (!this.deleteTargetId) return;
    
    this.isDeleting = true;
    this.cdr.detectChanges();
    
    // Show loading state on the row
    const runIndex = this.runs.findIndex(r => r._id === this.deleteTargetId);
    if (runIndex !== -1) {
      this.runs[runIndex]._deleting = true;
      this.cdr.detectChanges();
    }
    
    this.training.deleteTrainingRun(this.deleteTargetId).subscribe({
      next: (response: any) => {
        console.log('Delete response received:', response);
        console.log('Response type:', typeof response);
        console.log('Response deleted:', response.deleted);
        
        // Show cleanup results if available
        if (response.cleanup_results) {
          const results = response.cleanup_results;
          const successCount = [results.mongodb_deleted, results.milvus_collection_dropped, results.minio_bucket_removed].filter(Boolean).length;
          const totalCount = 3;
          
          if (successCount === totalCount) {
            console.log('Complete cleanup successful');
          } else {
            console.warn(`Partial cleanup: ${successCount}/${totalCount} operations succeeded`);
            if (results.errors && results.errors.length > 0) {
              console.error('Cleanup errors:', results.errors);
            }
          }
        }
        
        console.log('Before removing row, current runs count:', this.runs.length);
        console.log('Removing run with id:', this.deleteTargetId);
        
        // Remove from current page list
        this.runs = this.runs.filter(r => r._id !== this.deleteTargetId);
        this.total = Math.max(0, this.total - 1);
        
        console.log('After removing row, current runs count:', this.runs.length);
        
        // Close modal and reset state
        this.closeDeleteModal();
        
        // Trigger change detection to update the UI
        this.cdr.detectChanges();
        console.log('Change detection triggered');
        
        // Optionally fetch next item to fill the page
        if (this.page === 1) {
          this.training.listTrainingRuns(1, this.pageSize).subscribe({
            next: (res) => {
              this.runs = res.items || this.runs;
              this.total = res.total || this.total;
              this.cdr.detectChanges();
            },
            error: () => {}
          });
        }
      },
      error: (error) => {
        console.error('Delete failed:', error);
        // Reset deleting state
        const runIndex = this.runs.findIndex(r => r._id === this.deleteTargetId);
        if (runIndex !== -1) {
          this.runs[runIndex]._deleting = false;
        }
        // Close modal and reset state
        this.closeDeleteModal();
        this.cdr.detectChanges();
      },
      complete: () => {
        console.log('Delete observable completed');
      }
    });
  }

  onDeleteCancel(): void {
    this.closeDeleteModal();
  }

  private closeDeleteModal(): void {
    this.showDeleteModal = false;
    this.deleteTargetId = null;
    this.isDeleting = false;
  }

  private throttledDetectChanges(): void {
    const now = Date.now();
    if (now - this.lastUpdateTime >= this.UPDATE_THROTTLE_MS) {
      this.lastUpdateTime = now;
      this.cdr.detectChanges();
    }
  }
}
