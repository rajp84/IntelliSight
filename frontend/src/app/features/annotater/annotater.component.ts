import { Component, ElementRef, HostListener, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, Router } from '@angular/router';
import { SystemService } from '../../core/services/system.service';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-annotater',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './annotater.component.html',
  styleUrl: './annotater.component.scss'
})
export class AnnotaterComponent {
  videoPath: string = '';
  videoUrl: string = '';
  videoMime: string | null = null;
  runId: string | null = null;
  infoMessage: string = '';
  crops: Array<{ id: string; imageId: string; label: string; frameIndex: number; url: string }> = [];

  // Frame state
  frameIndex = 0;
  totalFrames = 0;
  fps = 0;
  playing = false;
  isPlaying = false;
  currentTime = 0;
  duration = 0;

  // Drawing state
  @ViewChild('canvas', { static: true }) canvasRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('video', { static: true }) videoRef!: ElementRef<HTMLVideoElement>;
  dragging = false;
  startX = 0;
  startY = 0;
  endX = 0;
  endY = 0;
  labelText = '';
  crossX: number | null = null;
  crossY: number | null = null;
  isSaving = false;

  // Background save queue
  saveQueue: Array<{ blob: Blob; frameIndex: number; bbox: { x1: number; y1: number; x2: number; y2: number }; label: string; runId?: string | null } > = [];
  processingQueue = false;

  constructor(private readonly route: ActivatedRoute,
              private readonly router: Router,
              private readonly system: SystemService,
              private readonly http: HttpClient) {}

  ngOnInit(): void {
    const p = this.route.snapshot.queryParamMap.get('path');
    if (!p) {
      this.router.navigateByUrl('/library');
      return;
    }
    this.videoPath = p;
    this.videoUrl = this.system.getLibraryFileUrl(this.videoPath);
    const ext = (this.videoPath.split('.').pop() || '').toLowerCase();
    const mimeByExt: Record<string, string> = {
      mp4: 'video/mp4',
      webm: 'video/webm',
      ogg: 'video/ogg',
      mov: 'video/quicktime',
      mkv: 'video/webm',
      avi: 'video/mp4',
    };
    this.videoMime = mimeByExt[ext] || null;
    // Fetch metadata and ensure a manual run is started
    this.http.get<{ fps: number; total_frames: number; width: number; height: number }>(`/api/train/manual/metadata`, { params: { path: this.videoPath } as any }).subscribe({
      next: (meta) => {
        this.fps = meta.fps || 30;
        this.totalFrames = meta.total_frames || 0;
      },
      error: () => {}
    });
    this.http.post<{ run_id: string }>(`/api/train/manual/start`, null, { params: { path: this.videoPath } as any }).subscribe({
      next: (res) => { this.runId = res?.run_id || null; if (this.runId) { this.loadCrops(); } },
      error: () => {}
    });
  }

  goBack(): void {
    this.router.navigateByUrl('/library');
  }

  ngAfterViewInit(): void {
    const canvas = this.canvasRef.nativeElement;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    this.redraw();
  }

  private get ctx(): CanvasRenderingContext2D | null {
    return this.canvasRef.nativeElement.getContext('2d');
  }

  onLoadedMetadata(): void {
    const v = this.videoRef.nativeElement;
    this.duration = v.duration || 0;
    // Derive fps if available
    if (!this.fps || !isFinite(this.fps)) {
      // try estimate using video frames if browser exposes it, else keep default
      this.fps = 30;
    }
  }

  onTimeUpdate(): void {
    const v = this.videoRef.nativeElement;
    this.currentTime = v.currentTime || 0;
  }

  togglePlayPause(): void {
    const v = this.videoRef.nativeElement;
    if (v.paused) {
      v.play();
      this.isPlaying = true;
    } else {
      v.pause();
      this.isPlaying = false;
    }
  }

  onSeek(event: Event): void {
    const input = event.target as HTMLInputElement;
    const val = Number(input.value);
    if (!this.duration || !isFinite(this.duration)) return;
    const t = (val / 1000) * this.duration;
    const v = this.videoRef.nativeElement;
    v.currentTime = Math.max(0, Math.min(this.duration, t));
    this.redraw();
  }

  formatTime(seconds: number): string {
    if (!seconds || seconds < 0 || !isFinite(seconds)) return '0:00';
    const s = Math.floor(seconds);
    const m = Math.floor(s / 60);
    const sec = s % 60;
    const pad = (n: number) => n.toString().padStart(2, '0');
    return `${m}:${pad(sec)}`;
  }

  private loadCrops(): void {
    if (!this.runId) return;
    this.http.get<{ items: Array<{ id: string; payload: any }> }>(`/api/train/runs/${encodeURIComponent(this.runId)}/embeddings`, { params: { limit: 200 } as any }).subscribe({
      next: (res) => {
        const items = res?.items || [];
        this.crops = items.map((it) => {
          const payload = it.payload || {};
          const imageId = payload.image_id || '';
          const label = payload.label || '';
          const frameIndex = typeof payload.frame_index === 'number' ? payload.frame_index : 0;
          const url = imageId ? `/api/train/runs/${encodeURIComponent(this.runId!)}/image/${encodeURIComponent(imageId)}` : '';
          return { id: it.id, imageId, label, frameIndex, url };
        }).filter(x => !!x.url);
      },
      error: () => {}
    });
  }

  // Video controls
  seekByFrames(delta: number): void {
    const v = this.videoRef.nativeElement;
    if (!this.fps || !isFinite(this.fps) || this.fps <= 0) {
      // approximate 30 fps
      this.fps = 30;
    }
    v.currentTime = Math.max(0, v.currentTime + delta / this.fps);
    this.redraw();
  }

  // Drawing handlers
  onPointerDown(event: PointerEvent): void {
    const rect = this.canvasRef.nativeElement.getBoundingClientRect();
    this.dragging = true;
    this.startX = event.clientX - rect.left;
    this.startY = event.clientY - rect.top;
    this.endX = this.startX;
    this.endY = this.startY;
    this.redraw();
  }

  onPointerMove(event: PointerEvent): void {
    const rect = this.canvasRef.nativeElement.getBoundingClientRect();
    const cx = event.clientX - rect.left;
    const cy = event.clientY - rect.top;
    this.crossX = Math.max(0, Math.min(this.canvasRef.nativeElement.width, cx));
    this.crossY = Math.max(0, Math.min(this.canvasRef.nativeElement.height, cy));
    if (this.dragging) {
      this.endX = cx;
      this.endY = cy;
    }
    this.redraw();
  }

  onPointerUp(): void {
    this.dragging = false;
    this.redraw();
  }

  onPointerLeave(): void {
    this.crossX = null;
    this.crossY = null;
    this.redraw();
  }

  // Crop current bbox and enqueue for background save with label
  cropAndLabel(): void {
    if (!this.labelText.trim()) return;
    const v = this.videoRef.nativeElement;
    const canvas = document.createElement('canvas');
    const vw = v.videoWidth;
    const vh = v.videoHeight;
    if (!vw || !vh) return;
    // Map canvas coords to video pixels
    const drawCanvas = this.canvasRef.nativeElement;
    const sx = Math.min(this.startX, this.endX);
    const sy = Math.min(this.startY, this.endY);
    const sw = Math.abs(this.endX - this.startX);
    const sh = Math.abs(this.endY - this.startY);
    if (sw < 2 || sh < 2) return;
    const scaleX = vw / drawCanvas.width;
    const scaleY = vh / drawCanvas.height;
    const vx = Math.max(0, Math.min(vw - 1, Math.round(sx * scaleX)));
    const vy = Math.max(0, Math.min(vh - 1, Math.round(sy * scaleY)));
    const vwCrop = Math.max(1, Math.round(sw * scaleX));
    const vhCrop = Math.max(1, Math.round(sh * scaleY));

    canvas.width = vwCrop;
    canvas.height = vhCrop;
    const cctx = canvas.getContext('2d');
    if (!cctx) return;
    cctx.drawImage(v, vx, vy, vwCrop, vhCrop, 0, 0, vwCrop, vhCrop);

    canvas.toBlob((blob) => {
      if (!blob) return;
      // Enqueue item
      const frameIdx = Math.round(v.currentTime * (this.fps || 30));
      this.saveQueue.push({
        blob,
        frameIndex: frameIdx,
        bbox: { x1: vx, y1: vy, x2: vx + vwCrop, y2: vy + vhCrop },
        label: this.labelText.trim(),
        runId: this.runId
      });
      // Keep label for faster repeated entry; just show info
      this.infoMessage = 'Queued crop for saving';
      setTimeout(() => { if (this.infoMessage === 'Queued crop for saving') this.infoMessage = ''; }, 1200);
      // Kick background processor
      this.processQueueIfNeeded();
    }, 'image/jpeg', 0.92);
  }

  completeRun(): void {
    if (!this.runId) return;
    this.http.post(`/api/train/manual/complete`, null, { params: { run_id: this.runId } as any }).subscribe({
      next: () => {
        this.infoMessage = 'Run completed';
        setTimeout(() => {
          this.infoMessage = '';
          this.router.navigateByUrl('/results');
        }, 500);
      },
      error: (err) => { console.error(err); }
    });
  }

  // Redraw overlay
  redraw(): void {
    const ctx = this.ctx;
    const canvas = this.canvasRef.nativeElement;
    const videoEl = this.videoRef.nativeElement;
    if (!ctx) return;
    // Resize canvas to video element size
    canvas.width = videoEl.clientWidth || 640;
    canvas.height = videoEl.clientHeight || 360;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Draw current rectangle
    const x = Math.min(this.startX, this.endX);
    const y = Math.min(this.startY, this.endY);
    const w = Math.abs(this.endX - this.startX);
    const h = Math.abs(this.endY - this.startY);
    if (w > 1 && h > 1) {
      ctx.strokeStyle = '#22c55e';
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);
    }

    // Draw crosshair if available
    if (this.crossX !== null && this.crossY !== null) {
      ctx.save();
      ctx.strokeStyle = 'rgba(255,255,255,0.6)';
      ctx.lineWidth = 1;
      // vertical
      ctx.beginPath();
      ctx.moveTo(this.crossX, 0);
      ctx.lineTo(this.crossX, canvas.height);
      ctx.stroke();
      // horizontal
      ctx.beginPath();
      ctx.moveTo(0, this.crossY);
      ctx.lineTo(canvas.width, this.crossY);
      ctx.stroke();
      ctx.restore();
    }
  }

  private processQueueIfNeeded(): void {
    if (this.processingQueue) return;
    if (this.saveQueue.length === 0) return;
    this.processingQueue = true;
    const next = () => {
      const item = this.saveQueue.shift();
      if (!item) {
        this.processingQueue = false;
        return;
      }
      const form = new FormData();
      form.append('image', item.blob, 'crop.jpg');
      form.append('path', this.videoPath);
      form.append('frame_index', String(item.frameIndex));
      form.append('bbox', JSON.stringify(item.bbox));
      form.append('label', item.label);
      if (item.runId) form.append('run_id', item.runId);
      this.http.post('/api/train/manual/crop', form).subscribe({
        next: (resp: any) => {
          if (resp?.run_id) this.runId = resp.run_id;
          this.loadCrops();
          // Continue with next item
          next();
        },
        error: (err) => {
          console.error('Failed to save crop', err);
          // Drop this item and continue
          next();
        }
      });
    };
    next();
  }

  // Keyboard shortcuts
  @HostListener('window:keydown', ['$event'])
  onKey(e: KeyboardEvent) {
    if (e.key === 'ArrowRight' && (e.shiftKey || e.altKey)) {
      this.seekByFrames(10);
      e.preventDefault();
    } else if (e.key === 'ArrowLeft' && (e.shiftKey || e.altKey)) {
      this.seekByFrames(-10);
      e.preventDefault();
    } else if (e.key === 'ArrowRight') {
      this.seekByFrames(1);
      e.preventDefault();
    } else if (e.key === 'ArrowLeft') {
      this.seekByFrames(-1);
      e.preventDefault();
    }
  }
}
