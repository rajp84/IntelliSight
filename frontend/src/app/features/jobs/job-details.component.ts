import { Component, OnInit, ViewChild, ElementRef, ViewChildren, QueryList, inject } from '@angular/core';
import { CommonModule, NgFor } from '@angular/common';
import { ActivatedRoute } from '@angular/router';
import { JobsService } from '../../core/services/jobs.service';

@Component({
  selector: 'app-job-details',
  standalone: true,
  imports: [CommonModule, NgFor],
  template: `
  <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
    <div>
      <div class="relative">
        <video #video el controls class="w-full bg-black" (loadedmetadata)="onVideoMeta()" (timeupdate)="onTimeUpdate()"></video>
        <canvas #overlay class="absolute left-0 top-0 pointer-events-none"></canvas>
      </div>
    </div>
    <div class="space-y-3">
      <div class="flex items-center justify-between">
        <h2 class="text-lg font-semibold">Job {{ jobId }}</h2>
        <button class="px-2 py-1 text-xs rounded bg-blue-600 text-white" (click)="loadMore()" [disabled]="loading">Load More</button>
      </div>
      <div #list class="h-[70vh] overflow-auto border rounded">
        <div *ngFor="let f of frames" #rowRef class="p-2 border-b hover:bg-gray-50 cursor-pointer" (click)="seekToFrame(f)" [attr.data-frame]="f.i"
             [class.bg-yellow-50]="f.i === currentFrame" [class.ring-1]="f.i === currentFrame" [class.ring-yellow-400]="f.i === currentFrame">
          <div class="text-xs text-gray-600">Frame {{ f.i }} | {{ f.t }} ms | {{ f.o?.length || 0 }} dets</div>
        </div>
      </div>
    </div>
  </div>
  `,
})
export class JobDetailsComponent implements OnInit {
  private readonly route = inject(ActivatedRoute);
  private readonly jobs = inject(JobsService);

  jobId: string = '';
  frames: any[] = [];
  afterIdx = -1;
  loading = false;
  job: any = null;
  currentFrame = -1;

  @ViewChild('video', { static: true }) videoRef!: ElementRef<HTMLVideoElement>;
  @ViewChild('overlay', { static: true }) canvasRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('list', { static: true }) listRef!: ElementRef<HTMLDivElement>;
  @ViewChildren('rowRef') rowRefs!: QueryList<ElementRef<HTMLDivElement>>;

  ngOnInit(): void {
    this.jobId = this.route.snapshot.paramMap.get('jobId') || '';
    this.jobs.getJob(this.jobId).subscribe(job => {
      this.job = job;
      const video = this.videoRef.nativeElement;
      if (this.job?.job_id) {
        // Stream via backend endpoint that supports range requests
        video.src = `/api/process/jobs/${encodeURIComponent(this.job.job_id)}/media`;
      }
    });
    this.loadMore();
  }

  loadMore(): void {
    if (this.loading || !this.jobId) return;
    this.loading = true;
    this.jobs.getJobFrames(this.jobId, this.afterIdx, 3).subscribe({
      next: (res) => {
        const newFrames = res.frames || [];
        if (newFrames.length) {
          this.frames = [...this.frames, ...newFrames];
          this.afterIdx = Number(res.last_idx ?? this.afterIdx);
        }
      },
      error: () => {},
      complete: () => { this.loading = false; }
    });
  }

  onVideoMeta(): void {
    const video = this.videoRef.nativeElement;
    const canvas = this.canvasRef.nativeElement;
    this.syncCanvasSize();
  }

  seekToFrame(f: any): void {
    const video = this.videoRef.nativeElement;
    if (!video || !this.job) return;
    // Prefer timestamp t (ms) to sync; fallback to fps/index if t missing
    let timeSec = Number(f?.t ?? 0) / 1000;
    if (!isFinite(timeSec) || timeSec < 0) {
      const fps = Number(this.job?.fps || 0);
      timeSec = fps > 0 ? (Number(f?.i || 0) / fps) : 0;
    }
    try {
      video.currentTime = timeSec;
    } catch {}
    setTimeout(() => this.drawDetections(f), 30);
    this.currentFrame = Number(f?.i || -1);
    this.scrollIntoView(this.currentFrame);
  }

  private drawDetections(f: any): void {
    const canvas = this.canvasRef.nativeElement;
    const video = this.videoRef.nativeElement;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    this.syncCanvasSize();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const dets: any[] = Array.isArray(f?.o) ? f.o : [];
    for (const d of dets) {
      const b = d?.b || [];
      if (b.length === 4) {
        const x1 = Math.round((b[0] || 0) * canvas.width);
        const y1 = Math.round((b[1] || 0) * canvas.height);
        const x2 = Math.round((b[2] || 0) * canvas.width);
        const y2 = Math.round((b[3] || 0) * canvas.height);
        ctx.strokeStyle = '#22c55e';
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, Math.max(1, x2 - x1), Math.max(1, y2 - y1));
        const label = String(d?.c ?? '');
        if (label) {
          ctx.fillStyle = '#22c55e';
          ctx.font = '12px sans-serif';
          const tw = ctx.measureText(label).width;
          ctx.fillRect(x1, Math.max(0, y1 - 16), tw + 6, 16);
          ctx.fillStyle = '#000000';
          ctx.fillText(label, x1 + 3, Math.max(12, y1 - 4));
        }
      }
    }
  }

  private syncCanvasSize(): void {
    const video = this.videoRef.nativeElement;
    const canvas = this.canvasRef.nativeElement;
    const w = Math.max(1, Math.floor(video.clientWidth || video.videoWidth || 0));
    const h = Math.max(1, Math.floor(video.clientHeight || video.videoHeight || 0));
    // Set drawing buffer size
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;
    // Ensure CSS size matches the video element box
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
  }

  onTimeUpdate(): void {
    const video = this.videoRef.nativeElement;
    const curSec = Number(video.currentTime || 0);
    const curMs = Math.round(curSec * 1000);
    // Find the most recent frame whose timestamp t <= current playback time
    let f: any | null = null;
    for (let i = this.frames.length - 1; i >= 0; i--) {
      const fr = this.frames[i];
      if (Number(fr?.t || 0) <= curMs) { f = fr; break; }
    }
    if (!f) return;
    const frameIdx = Number(f?.i ?? -1);
    if (frameIdx !== this.currentFrame) {
      this.currentFrame = frameIdx;
      this.drawDetections(f);
      this.scrollIntoView(frameIdx);
    }
  }

  private scrollIntoView(frameIdx: number): void {
    const container = this.listRef?.nativeElement;
    if (!container) return;
    // Find row element by matching its inner text or index mapping
    let rowEl: HTMLDivElement | null = null;
    if (this.rowRefs && this.rowRefs.length) {
      const match = this.rowRefs.find((ref: ElementRef<HTMLDivElement>) => {
        const el = ref.nativeElement;
        const val = el.getAttribute('data-frame');
        return val !== null && Number(val) === frameIdx;
      });
      rowEl = match ? match.nativeElement : null;
    }
    if (rowEl) {
      const cRect = container.getBoundingClientRect();
      const rRect = rowEl.getBoundingClientRect();
      if (rRect.top < cRect.top || rRect.bottom > cRect.bottom) {
        container.scrollTop += (rRect.top - cRect.top) - (cRect.height / 3);
      }
    }
  }
}


