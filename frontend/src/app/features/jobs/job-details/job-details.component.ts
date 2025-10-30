import { Component, OnInit, ViewChild, ElementRef, ViewChildren, QueryList, inject } from '@angular/core';
import { CommonModule, NgFor } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { CropEditorComponent } from '../../../shared/components/crop-editor.component';
import { ActivatedRoute } from '@angular/router';
import { JobsService } from '../../../core/services/jobs.service';

@Component({
  selector: 'app-job-details',
  standalone: true,
  imports: [CommonModule, NgFor, FormsModule, CropEditorComponent],
  templateUrl: './job-details.component.html',
  styleUrl: './job-details.component.scss'
})
export class JobDetailsComponent implements OnInit {
  private readonly route = inject(ActivatedRoute);
  private readonly jobs = inject(JobsService);

  jobId: string = '';
  frames: any[] = [];
  afterIdx = -1;
  loading = false;
  job: any = null;
  mediaSrc: string = '';
  conf: number = 0.6;
  currentFrame = -1;
  private prefetchThresholdFrames = 128;
  private isPlaying = false;
  private programmaticScroll = false;
  private rowHeightPx = 0;
  private userScrolling = false;
  private userScrollResetTimer: any = null;
  private rafId: number | null = null;
  private lastDrawnIdx: number = -1;
  private lastScrollIdx: number = -1;
  private lastScrollAt: number = 0;
  private smoothScrollThresholdPx: number = 16; // hysteresis to reduce jitter
  private desiredScrollTop: number | null = null;
  private scrollSmoothness: number = 0.75; // 0..1, higher = faster
  currentTracks: Array<{ k: number; leftPct: number; widthPct: number; label: string; color: string; durationMs: number; display: string; fromMs: number; toMs: number; segments?: Array<{ leftPct: number; widthPct: number; fromMs: number; toMs: number }> }> = [];
  playheadLeftPct = 0;
  private trackRowHeight = 14;
  // Crop editor state
  showCrop = false;
  cropImageUrl: string | null = null;
  cropBox: { x1: number; y1: number; x2: number; y2: number } | null = null;
  cropFrame: any = null;
  cropLabel: string = '';
  savingCrop = false;

  @ViewChild('video', { static: false }) videoRef!: ElementRef<HTMLVideoElement>;
  @ViewChild('image', { static: false }) imageRef!: ElementRef<HTMLImageElement>;
  @ViewChild('overlay', { static: true }) canvasRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('list', { static: true }) listRef!: ElementRef<HTMLDivElement>;
  @ViewChildren('rowRef') rowRefs!: QueryList<ElementRef<HTMLDivElement>>;

  ngOnInit(): void {
    this.jobId = this.route.snapshot.paramMap.get('jobId') || '';
    this.jobs.getJob(this.jobId).subscribe(job => {
      this.job = job;
      const jconf = Number(job?.conf);
      if (isFinite(jconf) && jconf >= 0 && jconf <= 1) this.conf = jconf;
      if (this.job?.job_id) {
        this.mediaSrc = `/api/process/jobs/${encodeURIComponent(this.job.job_id)}/media`;
      }
    });
    this.loadMore();
  }

  onConfChange(ev: Event): void {
    const input = ev.target as HTMLInputElement;
    const val = Number(input?.value);
    if (!isFinite(val)) return;
    const clamped = Math.max(0, Math.min(1, val));
    this.conf = clamped;
    // Re-render current frame and rebuild timeline with filter
    if (this.currentFrame >= 0) {
      const curFrame = this.frames.find(fr => Number(fr?.i || -1) === this.currentFrame);
      if (curFrame) this.drawDetections(curFrame);
    }
    this.buildAllTracks();
  }

  get isImage(): boolean {
    const tf = Number(this.job?.total_frames || 0);
    const fps = Number(this.job?.fps || 0);
    return tf === 1 || fps <= 0;
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
          // Rebuild full timeline from all loaded frames (prepopulate)
          this.buildAllTracks();
        }
      },
      error: () => {},
      complete: () => { this.loading = false; }
    });
  }

  onVideoMeta(): void {
    this.syncCanvasSize();
  }

  onPlay(): void { this.isPlaying = true; }
  onPause(): void { this.isPlaying = false; }

  seekToFrame(f: any): void {
    if (!this.job) return;
    if (!this.isImage) {
      const video = this.videoRef?.nativeElement;
      if (!video) return;
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
    } else {
      // Image: just draw detections for the single frame
      setTimeout(() => this.drawDetections(f), 0);
    }
    this.currentFrame = Number(f?.i || -1);
    this.scrollIntoView(this.currentFrame);
  }

  private drawDetections(f: any): void {
    const canvas = this.canvasRef.nativeElement;
    const video = this.videoRef?.nativeElement;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    this.syncCanvasSize();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (this.job?.type === 'similarity_search') {
      // No bbox drawing for similarity_search
      return;
    }
    const dets: any[] = (Array.isArray(f?.o) ? f.o : []).filter((d: any) => {
      const sc = Number(d?.s);
      return !isFinite(sc) || sc >= this.conf;
    });
    for (const d of dets) {
      const b = d?.b || [];
      if (b.length === 4) {
        const x1 = Math.round((b[0] || 0) * canvas.width);
        const y1 = Math.round((b[1] || 0) * canvas.height);
        const x2 = Math.round((b[2] || 0) * canvas.width);
        const y2 = Math.round((b[3] || 0) * canvas.height);
        // Color by confidence score
        const sc = Number(d?.s);
        let col = '#ef4444'; // red (low)
        if (isFinite(sc)) {
          if (sc >= 0.80) col = '#22c55e';        // green (high)
          else if (sc >= 0.65) col = '#a7dfb6';  // light green (med-high)
          else if (sc >= 0.45) col = '#eab308';  // yellow (medium)
          else if (sc >= 0.30) col = '#f97316';  // orange (low-medium)
          else col = '#ef4444';                  // red (low)
        }
        ctx.strokeStyle = col;
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, Math.max(1, x2 - x1), Math.max(1, y2 - y1));
        const cls = String(d?.c ?? '');
        const pct = isFinite(sc) && sc >= 0 ? `${(sc * 100).toFixed(1)}%` : '';
        const label = cls && pct ? `${cls} ${pct}` : (cls || pct);
        if (label) {
          ctx.fillStyle = col;
          ctx.font = '12px sans-serif';
          const tw = ctx.measureText(label).width;
          ctx.fillRect(x1, Math.max(0, y1 - 16), tw + 6, 16);
          ctx.fillStyle = '#000000';
          ctx.fillText(label, x1 + 3, Math.max(12, y1 - 4));
        }
      }
    }
  }

  onImageLoad(): void {
    // When the image element finishes loading, size the overlay and render first frame
    this.syncCanvasSize();
    const first = this.frames?.[0];
    if (first) {
      this.currentFrame = Number(first?.i || -1);
      this.drawDetections(first);
    }
  }

  similarityItems(f: any): Array<{ label: string; score: number }> {
    if (!f) return [];
    const items = Array.isArray(f?.o) ? f.o : [];
    const mapped = items.map((it: any) => {
      const lbl = String(it?.m?.label ?? it?.label ?? it?.ref ?? 'item');
      const score = Number(it?.s ?? it?.score ?? 0);
      return { label: lbl, score: isFinite(score) ? score : 0 };
    });
    // Deduplicate by label, keep highest score
    const best: Record<string, number> = {};
    for (const m of mapped) {
      if (!(m.label in best) || m.score > best[m.label]) best[m.label] = m.score;
    }
    return Object.entries(best).map(([label, score]) => ({ label, score }));
  }

  get currentSimilarityResults(): Array<{ label: string; score: number; thumbUrl?: string }>{
    if (this.job?.type !== 'similarity_search') return [];
    const f = this.frames.find(fr => Number(fr?.i || -1) === this.currentFrame);
    if (!f) return [];
    const items: any[] = Array.isArray(f?.o) ? f.o : [];
    return items.map((it: any) => {
      const lbl = String(it?.m?.label ?? it?.label ?? it?.ref ?? 'item');
      const score = Number(it?.s ?? it?.score ?? 0);
      const thumbUrl = this.extractThumbUrl(it);
      return { label: lbl, score: isFinite(score) ? score : 0, thumbUrl };
    });
  }

  private extractThumbUrl(item: any): string | undefined {
    if (!item) return undefined;
    const meta = item?.m;
    // Prefer MinIO things image via image_id if available
    const imageId = item?.img_id || meta?.image_id || meta?.id;
    if (typeof imageId === 'string' && imageId) {
      const filename = encodeURIComponent(`${imageId}.jpg`);
      return `/api/things/image/${filename}`;
    }
    if (!meta) return undefined;
    // Prefer explicit URLs
    const url = meta.thumbnail_url || meta.thumb_url || meta.image_url || meta.url;
    if (typeof url === 'string' && url) return url;
    // Support base64 thumbnail if provided
    const b64 = meta.thumbnail_b64 || meta.thumb_b64 || meta.image_b64 || meta.b64;
    if (typeof b64 === 'string' && b64) {
      // Heuristic: if already prefixed, return as-is
      if (b64.startsWith('data:image')) return b64;
      return `data:image/jpeg;base64,${b64}`;
    }
    return undefined;
  }

  private syncCanvasSize(): void {
    const mediaEl = this.isImage ? (this.imageRef?.nativeElement as HTMLElement | undefined) : (this.videoRef?.nativeElement as HTMLElement | undefined);
    const canvas = this.canvasRef.nativeElement;
    if (!mediaEl) return;
    // Prefer client box; fallback to intrinsic sizes
    const anyEl: any = mediaEl as any;
    const w = Math.max(1, Math.floor(mediaEl.clientWidth || anyEl.videoWidth || anyEl.naturalWidth || 0));
    const h = Math.max(1, Math.floor(mediaEl.clientHeight || anyEl.videoHeight || anyEl.naturalHeight || 0));
    // Set drawing buffer size
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;
    // Ensure CSS size matches the video element box
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
  }

  onTimeUpdate(): void {
    // Kick off a high-frequency render loop for smooth overlays
    if (this.isPlaying && this.rafId === null) {
      this.startRenderLoop();
    }
  }

  private scrollIntoView(frameIdx: number): void {
    const container = this.listRef?.nativeElement;
    if (!container) return;
    // Prefer exact DOM position of the target row to avoid drift with variable heights
    const rowEl = container.querySelector<HTMLDivElement>(`[data-frame="${frameIdx}"]`);
    if (rowEl) {
      const containerHeight = container.clientHeight || 0;
      const rowH = rowEl.offsetHeight || this.rowHeightPx || 36;
      const rowTop = rowEl.offsetTop || 0; // offsetTop relative to scroll container
      const desired = Math.max(0, rowTop - (containerHeight / 2 - rowH / 2));
      const maxScroll = Math.max(0, container.scrollHeight - containerHeight);
      const clamped = Math.max(0, Math.min(maxScroll, desired));
      const delta = Math.abs((container.scrollTop || 0) - clamped);
      if (delta > this.smoothScrollThresholdPx) {
        this.programmaticScroll = true;
        this.desiredScrollTop = clamped;
      }
      return;
    }
    // Fallback: estimate from row index if element not found yet
    this.ensureRowHeight();
    const rowIndex = this.frames.findIndex(fr => Number(fr?.i || -1) === frameIdx);
    if (rowIndex < 0) return;
    const containerHeight = container.clientHeight || 0;
    const rowH = this.rowHeightPx || 36;
    let desired = rowIndex * rowH - (containerHeight / 2 - rowH / 2);
    desired = Math.max(0, Math.round(desired / rowH) * rowH);
    const maxScroll = Math.max(0, container.scrollHeight - containerHeight);
    const clamped = Math.max(0, Math.min(maxScroll, desired));
    const delta = Math.abs((container.scrollTop || 0) - clamped);
    if (delta > this.smoothScrollThresholdPx) {
      this.programmaticScroll = true;
      this.desiredScrollTop = clamped;
    }
  }

  onListScroll(): void {
    const container = this.listRef?.nativeElement;
    if (!container || this.loading) return;
    if (this.programmaticScroll) return;
    const remaining = container.scrollHeight - (container.scrollTop + container.clientHeight);
    // If close to bottom, prefetch more
    if (remaining < 400) {
      this.loadMore();
    }
    // Pause and seek video to highlighted (center) row
    if (this.userScrolling) {
      // Prefer DOM-based center row detection to avoid drift with variable heights
      const rect = container.getBoundingClientRect();
      const cx = Math.floor(rect.left + (container.clientWidth / 2));
      const cy = Math.floor(rect.top + (container.clientHeight / 2));
      let hit = document.elementFromPoint(cx, cy) as HTMLElement | null;
      let rowEl: HTMLElement | null = hit;
      let frameAttr: string | null = null;
      while (rowEl) {
        if (rowEl.hasAttribute && rowEl.hasAttribute('data-frame')) {
          frameAttr = rowEl.getAttribute('data-frame');
          break;
        }
        if (rowEl === container) break;
        rowEl = rowEl.parentElement as HTMLElement | null;
      }
      let near: any | null = null;
      if (frameAttr != null) {
        const idxFromAttr = Number(frameAttr);
        near = this.frames.find(fr => Number(fr?.i || -1) === idxFromAttr) || null;
      }
      // Fallback: estimate by row height if DOM query fails
      if (!near) {
        this.ensureRowHeight();
        const approxIndex = Math.max(0, Math.floor((container.scrollTop + container.clientHeight / 2) / Math.max(1, this.rowHeightPx)));
        near = this.findNearestFrameByIndex(approxIndex);
      }
      if (near) {
        if (!this.isImage) {
          const video = this.videoRef?.nativeElement;
          if (video) {
            try { video.pause(); } catch {}
            // Seek by timestamp t (ms), fallback to fps/index
            let timeSec = Number(near.t ?? 0) / 1000;
            if (!isFinite(timeSec) || timeSec < 0) {
              const fps = Number(this.job?.fps || 0);
              timeSec = fps > 0 ? (Number(near.i || 0) / fps) : 0;
            }
            try { video.currentTime = timeSec; } catch {}
            this.currentFrame = Number(near.i || -1);
            this.drawDetections(near);
            const curMs = Math.round(timeSec * 1000);
            this.updatePlayhead(curMs);
            this.ensureTimelineTrackVisible(curMs);
          }
        } else {
          this.currentFrame = Number(near.i || -1);
          this.drawDetections(near);
          const curMs = Number(near.t ?? 0);
          this.updatePlayhead(isFinite(curMs) ? curMs : 0);
          this.ensureTimelineTrackVisible(isFinite(curMs) ? curMs : 0);
        }
      }
    }
  }

  private maybePrefetch(frameIdx: number): void {
    if (this.loading) return;
    if (this.afterIdx >= 0 && frameIdx >= (this.afterIdx - this.prefetchThresholdFrames)) {
      this.loadMore();
    }
  }

  private ensureRowHeight(): void {
    if (this.rowHeightPx > 0) return;
    const first = this.rowRefs?.first?.nativeElement;
    this.rowHeightPx = Math.max(24, Math.floor(first?.offsetHeight || 0) || 36);
  }

  private findNearestFrameByIndex(targetIdx: number): any | null {
    if (!this.frames || this.frames.length === 0) return null;
    // frames are appended in order; binary search by frame i
    let lo = 0, hi = this.frames.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const vi = Number(this.frames[mid]?.i || 0);
      if (vi === targetIdx) return this.frames[mid];
      if (vi < targetIdx) lo = mid + 1; else hi = mid - 1;
    }
    // hi is the largest < target, lo is smallest > target; pick nearest
    const candA = (hi >= 0) ? this.frames[hi] : null;
    const candB = (lo < this.frames.length) ? this.frames[lo] : null;
    if (candA && candB) {
      const da = Math.abs(Number(candA.i || 0) - targetIdx);
      const db = Math.abs(Number(candB.i || 0) - targetIdx);
      return da <= db ? candA : candB;
    }
    return candA || candB || null;
  }

  onUserScroll(): void {
    this.userScrolling = true;
    if (this.userScrollResetTimer) clearTimeout(this.userScrollResetTimer);
    this.userScrollResetTimer = setTimeout(() => { this.userScrolling = false; }, 200);
  }

  onUserKey(ev: KeyboardEvent): void {
    // Arrow keys / page up/down / home/end treated as user scroll intents
    const keys = ['ArrowUp','ArrowDown','PageUp','PageDown','Home','End'];
    if (keys.includes(ev.key)) {
      this.userScrolling = true;
    }
  }

  private startRenderLoop(): void {
    const render = () => {
      if (!this.isPlaying) {
        this.stopRenderLoop();
        return;
      }
      const video = this.videoRef.nativeElement;
      const curSec = Number(video.currentTime || 0);
      const curMs = Math.round(curSec * 1000);
      const f = this.findFrameByTime(curMs);
      if (f) {
        const idx = Number(f.i || -1);
        if (idx !== this.lastDrawnIdx) {
          this.lastDrawnIdx = idx;
          this.currentFrame = idx;
          this.drawDetections(f);
          // Prebuilt timeline; just update playhead and keep panes in sync
          this.updatePlayhead(curMs);
          this.ensureTimelineTrackVisible(curMs);
          // Throttle auto-centering to ~10Hz and only when needed
          const now = performance.now();
          if ((idx !== this.lastScrollIdx && now - this.lastScrollAt > 100) && !this.userScrolling) {
            this.scrollIntoView(idx);
            this.lastScrollIdx = idx;
            this.lastScrollAt = now;
          }
          this.maybePrefetch(idx);
        }
      }
      // Smooth scroll step if we have a target
      const container = this.listRef?.nativeElement;
      if (this.desiredScrollTop != null && container) {
        const cur = container.scrollTop || 0;
        const target = this.desiredScrollTop;
        const d = target - cur;
        if (Math.abs(d) <= 0.5) {
          container.scrollTop = target;
          this.desiredScrollTop = null;
          // Release the programmatic flag a bit later to ignore trailing events
          setTimeout(() => { this.programmaticScroll = false; }, 60);
        } else {
          container.scrollTop = cur + d * this.scrollSmoothness;
        }
      }
      this.rafId = requestAnimationFrame(render);
    };
    this.rafId = requestAnimationFrame(render);
  }

  private stopRenderLoop(): void {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
  }

  private findFrameByTime(curMs: number): any | null {
    if (!this.frames || this.frames.length === 0) return null;
    // Binary search by timestamp t
    let lo = 0, hi = this.frames.length - 1, ans = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const tm = Number(this.frames[mid]?.t || 0);
      if (tm <= curMs) { ans = mid; lo = mid + 1; } else { hi = mid - 1; }
    }
    if (ans >= 0) return this.frames[ans];
    return this.frames[0];
  }
  
  private buildAllTracks(): void {
    const fps = Number(this.job?.fps || 0);
    const totalFrames = Number(this.job?.total_frames || 0);
    const totalMs = fps > 0 ? (totalFrames / fps) * 1000.0 : (this.frames.length > 0 ? Number(this.frames[this.frames.length-1]?.t || 0) : 0);
    if (this.job?.type === 'similarity_search') {
      // Build contiguous segments per label to show gaps
      type Seg = { fromMs: number; toMs: number };
      const byLabel: Map<string, { open: Seg | null; segs: Seg[] }>=new Map();
      const getFrameTime = (fr: any, idx: number) => {
        const t = Number(fr?.t);
        if (isFinite(t) && t >= 0) return t;
        const fpsLocal = Number(this.job?.fps || 0);
        return fpsLocal > 0 ? Math.round((idx / fpsLocal) * 1000) : 0;
      };
      for (let i = 0; i < this.frames.length; i++) {
        const fr = this.frames[i];
        const tMs = getFrameTime(fr, i);
        const labelsPresent = new Set(this.similarityItems(fr).map(it => it.label));
        // Close segments for labels that vanished
        for (const [label, st] of byLabel.entries()) {
          if (st.open && !labelsPresent.has(label)) {
            st.open.toMs = tMs;
            st.segs.push(st.open);
            st.open = null;
          }
        }
        // Open/extend segments for labels present
        for (const label of labelsPresent) {
          let st = byLabel.get(label);
          if (!st) { st = { open: null, segs: [] }; byLabel.set(label, st); }
          if (!st.open) {
            st.open = { fromMs: tMs, toMs: tMs };
          } else {
            st.open.toMs = tMs;
          }
        }
      }
      // Close any remaining open segments at end
      const endMs = totalMs;
      for (const st of byLabel.values()) {
        if (st.open) {
          st.open.toMs = endMs;
          st.segs.push(st.open);
          st.open = null;
        }
      }
      const items: Array<{ k: number; leftPct: number; widthPct: number; label: string; color: string; durationMs: number; display: string; fromMs: number; toMs: number; segments?: Array<{ leftPct: number; widthPct: number; fromMs: number; toMs: number }> }> = [];
      const labelsSorted = Array.from(byLabel.keys()).sort((a,b)=>{
        const aStart = (byLabel.get(a)?.segs[0]?.fromMs ?? 0);
        const bStart = (byLabel.get(b)?.segs[0]?.fromMs ?? 0);
        return aStart - bStart;
      });
      for (const label of labelsSorted) {
        const st = byLabel.get(label)!;
        const segs = st.segs.filter(s => (s.toMs - s.fromMs) > 0).map(s => ({
          fromMs: s.fromMs,
          toMs: s.toMs,
          leftPct: totalMs > 0 ? Math.max(0, Math.min(100, (s.fromMs / totalMs) * 100)) : 0,
          widthPct: totalMs > 0 ? Math.max(0.5, Math.min(100, ((Math.max(1, s.toMs - s.fromMs)) / totalMs) * 100)) : 1,
        }));
        const firstFrom = segs.length ? segs[0].fromMs : 0;
        const lastTo = segs.length ? segs[segs.length - 1].toMs : 0;
        items.push({
          k: items.length,
          label,
          display: label,
          color: '#3b82f6',
          fromMs: firstFrom,
          toMs: lastTo,
          durationMs: Math.max(0, lastTo - firstFrom),
          leftPct: totalMs > 0 ? Math.max(0, Math.min(100, (firstFrom / totalMs) * 100)) : 0,
          widthPct: totalMs > 0 ? Math.max(0.5, Math.min(100, ((Math.max(1, lastTo - firstFrom)) / totalMs) * 100)) : 1,
          segments: segs,
        });
      }
      this.currentTracks = items;
      this.syncTimelineScroll();
      return;
    }
    type Span = { fromMs: number; toMs: number; firstFrame: number; label: string; peakScore: number };
    const spans = new Map<number, Span>();
    // Build spans from all loaded frames
    for (const fr of this.frames) {
      const fi = Number(fr?.i || 0);
      const ft = Number(fr?.t || 0);
      const dets: any[] = (Array.isArray(fr?.o) ? fr.o : []).filter((d: any) => {
        const sc = Number(d?.s);
        return !isFinite(sc) || sc >= this.conf;
      });
      for (const d of dets) {
        const k = Number(d?.k);
        if (!isFinite(k)) continue;
        const fromMs = Number(d?.from ?? ft);
        const toMs = Number(d?.to ?? ft);
        const label = String(d?.c ?? `ID ${k}`);
        const sc = Number(d?.s);
        const cur = spans.get(k);
        if (!cur) {
          spans.set(k, { fromMs, toMs, firstFrame: fi, label, peakScore: isFinite(sc) ? sc : -1 });
        } else {
          if (fromMs < cur.fromMs) cur.fromMs = fromMs;
          if (toMs > cur.toMs) cur.toMs = toMs;
          if (fi < cur.firstFrame) cur.firstFrame = fi;
          if (!cur.label && label) cur.label = label;
          if (isFinite(sc) && sc > cur.peakScore) cur.peakScore = sc;
        }
      }
    }
    // Convert spans to timeline bars in order of appearance
    const items: Array<{ k: number; leftPct: number; widthPct: number; label: string; color: string; durationMs: number; display: string; fromMs: number; toMs: number }> = [];
    const sorted = Array.from(spans.entries()).sort((a,b)=> a[1].firstFrame - b[1].firstFrame);
    for (const [k, s] of sorted) {
      const duration = Math.max(0, s.toMs - s.fromMs);
      const leftPct = totalMs > 0 ? Math.max(0, Math.min(100, (s.fromMs / totalMs) * 100)) : 0;
      const widthPct = totalMs > 0 ? Math.max(0.5, Math.min(100, ((Math.max(10, duration)) / totalMs) * 100)) : 1;
      const color = this.colorForScore(s.peakScore);
      const display = `${s.label} (ID ${k})`;
      items.push({ k, leftPct, widthPct, label: s.label, color, durationMs: duration, display, fromMs: s.fromMs, toMs: s.toMs });
    }
    this.currentTracks = items;
    this.syncTimelineScroll();
  }

  private colorForId(k: number): string {
    const colors = ['#60a5fa','#34d399','#fbbf24','#f472b6','#a78bfa','#f59e0b','#10b981','#3b82f6'];
    return colors[Math.abs(k) % colors.length];
  }

  private colorForScore(sc: number): string {
    let col = '#ef4444';
    if (isFinite(sc)) {
      if (sc >= 0.80) col = '#22c55e';
      else if (sc >= 0.65) col = '#a7dfb6';
      else if (sc >= 0.45) col = '#eab308';
      else if (sc >= 0.30) col = '#f97316';
      else col = '#ef4444';
    }
    return col;
  }

  private updatePlayhead(curMs: number): void {
    const fps = Number(this.job?.fps || 0);
    const totalFrames = Number(this.job?.total_frames || 0);
    const totalMs = fps > 0 ? (totalFrames / fps) * 1000.0 : (this.frames.length > 0 ? Number(this.frames[this.frames.length-1]?.t || 0) : 0);
    this.playheadLeftPct = totalMs > 0 ? Math.max(0, Math.min(100, (curMs / totalMs) * 100)) : 0;
  }

  // Keep labels and bars columns scrolled together
  @ViewChild('timelineLabels', { static: false }) timelineLabelsRef!: ElementRef<HTMLDivElement>;
  @ViewChild('timelineBars', { static: false }) timelineBarsRef!: ElementRef<HTMLDivElement>;

  onTimelineScroll(): void {
    const labels = this.timelineLabelsRef?.nativeElement;
    const bars = this.timelineBarsRef?.nativeElement;
    if (!labels || !bars) return;
    labels.scrollTop = bars.scrollTop;
  }

  private syncTimelineScroll(): void {
    const labels = this.timelineLabelsRef?.nativeElement;
    const bars = this.timelineBarsRef?.nativeElement;
    if (!labels || !bars) return;
    labels.scrollTop = bars.scrollTop;
  }

  private ensureTimelineTrackVisible(curMs: number): void {
    const bars = this.timelineBarsRef?.nativeElement;
    const labels = this.timelineLabelsRef?.nativeElement;
    if (!bars || !labels) return;
    const activeIdx = this.currentTracks.findIndex(t => curMs >= t.fromMs && curMs <= t.toMs);
    if (activeIdx < 0) return;
    const visibleH = bars.clientHeight || 96;
    const rowTop = activeIdx * this.trackRowHeight;
    const desired = Math.max(0, Math.floor(rowTop - (visibleH / 2 - this.trackRowHeight / 2)));
    const maxScroll = Math.max(0, bars.scrollHeight - visibleH);
    const clamped = Math.max(0, Math.min(maxScroll, desired));
    bars.scrollTop = clamped;
    labels.scrollTop = clamped;
  }

  get timelineHeightPx(): number {
    const rows = this.currentTracks.length;
    const h = 4 + rows * 14;
    return h < 96 ? 96 : h;
  }

  // --- Frame actions (detection only) ---
  toggleFrameInvalid(f: any): void {
    if (!this.job || !f) return;
    const jobId = String(this.job.job_id || '');
    const idx = Number(f.i || -1);
    if (!jobId || idx < 0) return;
    const invalid = !Boolean(f.invalid);
    fetch(`/api/process/jobs/${encodeURIComponent(jobId)}/frames/${idx}/invalid`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ invalid })
    }).then(() => {
      f.invalid = invalid;
    }).catch(() => {});
  }

  openFixCrop(f: any): void {
    if (!this.job || !f) return;
    const jobId = String(this.job.job_id || '');
    const idx = Number(f.i || -1);
    if (!jobId || idx < 0) return;
    // Get current frame image from streaming endpoint by seeking video element to frame time
    const video = this.videoRef.nativeElement;
    // Attempt to capture current canvas overlay merged with video frame? For now, use streaming video tag frame via canvas snapshot
    this.seekToFrame(f);
    setTimeout(() => {
      const canvas = document.createElement('canvas');
      const w = Math.max(1, Math.floor(video.videoWidth || video.clientWidth || 0));
      const h = Math.max(1, Math.floor(video.videoHeight || video.clientHeight || 0));
      canvas.width = w; canvas.height = h;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        try { ctx.drawImage(video, 0, 0, w, h); } catch {}
        this.cropImageUrl = canvas.toDataURL('image/jpeg', 0.92);
      }
      this.cropBox = null;
      this.cropFrame = { jobId, idx };
      this.showCrop = true;
    }, 150);
  }

  confirmCrop(): void {
    if (!this.showCrop || !this.cropImageUrl || !this.cropFrame || !this.cropBox || !this.cropLabel) { this.showCrop = false; return; }
    const { jobId, idx } = this.cropFrame as { jobId: string; idx: number };
    const { x1, y1, x2, y2 } = this.cropBox;
    this.savingCrop = true;
    fetch(`/api/process/jobs/${encodeURIComponent(jobId)}/frames/${idx}/add-to-things`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label: this.cropLabel, x1, y1, x2, y2 })
    }).then(() => {
      this.savingCrop = false;
      this.showCrop = false; this.cropLabel = '';
    }).catch(() => {
      this.savingCrop = false;
      this.showCrop = false; this.cropLabel = '';
    });
  }

  cancelCrop(): void { this.showCrop = false; }
}
