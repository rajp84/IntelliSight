import { Component, OnInit, OnDestroy, inject } from '@angular/core';
import { CommonModule, NgFor, NgIf, DatePipe } from '@angular/common';
import { RouterLink } from '@angular/router';
import { JobsService, JobItem } from '../../core/services/jobs.service';
import { SocketService } from '../../core/services/socket.service';

@Component({
  selector: 'app-jobs',
  standalone: true,
  imports: [CommonModule, NgIf, NgFor, DatePipe, RouterLink],
  templateUrl: './jobs.component.html',
})
export class JobsComponent implements OnInit, OnDestroy {
  private readonly jobsSvc = inject(JobsService);
  private readonly socket = inject(SocketService);
  loading = false;
  items: JobItem[] = [];
  deleting: Record<string, boolean> = {};
  private pollHandle: any;

  ngOnInit(): void {
    this.refresh();
    this.socket.connect();
    this.socket.on<any>('job_update', (msg) => {
      if (!msg || !msg.job_id) return;
      const idx = this.items.findIndex(x => x.job_id === msg.job_id);
      if (idx >= 0) {
        this.items[idx] = { ...this.items[idx], ...msg } as JobItem;
        this.items = [...this.items];
      } else {
        // If job not loaded yet, fetch latest list (best-effort)
        this.refresh();
      }
    });
    // Fallback polling to keep previews fresh if socket updates are missed
    // this.pollHandle = setInterval(() => {
    //   if (this.items.some(it => it.status === 'running')) {
    //     this.refresh();
    //   }
    // }, 2000);
  }

  ngOnDestroy(): void {
    if (this.pollHandle) {
      clearInterval(this.pollHandle);
      this.pollHandle = null;
    }
  }

  refresh(): void {
    if (this.loading) return;
    this.loading = true;
    this.jobsSvc.listJobs(200).subscribe({
      next: (res) => { this.items = res.items || []; },
      error: () => { this.items = []; },
      complete: () => { this.loading = false; }
    });
  }

  delete(j: JobItem): void {
    if (!j?.job_id || this.deleting[j.job_id]) return;
    if (!confirm('Delete this job and its frames?')) return;
    this.deleting[j.job_id] = true;
    this.jobsSvc.deleteJob(j.job_id).subscribe({
      next: () => {
        this.items = this.items.filter(it => it.job_id !== j.job_id);
        delete this.deleting[j.job_id];
      },
      error: () => {
        delete this.deleting[j.job_id];
      }
    });
  }
}


