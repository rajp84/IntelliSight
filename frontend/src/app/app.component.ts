import { Component, OnInit } from '@angular/core';
import { NgIf, NgFor } from '@angular/common';
import { RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';
import { SocketService } from './core/services/socket.service';
import { SystemService } from './core/services/system.service';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, RouterLink, RouterLinkActive, NgIf, NgFor],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent implements OnInit {
  title = 'IntelliSight';
  currentYear = new Date().getFullYear();
  tooltipVisible = false;
  tooltipText = '';
  tooltipTop = 0;
  tooltipLeft = 0;

  cpuLabel = '';
  memLabel = '';
  diskLabel = '';
  gpuLabels: string[] = [];
  trainingThumb: string | null = null;
  trainingFooter: { state: string; progress: number; message?: string; queue_detect?: number; queue_detect_max?: number; queue_results?: number; fps?: number } | null = null;
  terminating = false;

  showTooltip(event: MouseEvent, text: string): void {
    const target = event.currentTarget as HTMLElement;
    const rect = target.getBoundingClientRect();
    this.tooltipTop = rect.top + rect.height / 2 - 16;
    this.tooltipLeft = rect.right + 4; // account for arrow overhang so tip touches icon
    this.tooltipText = text;
    this.tooltipVisible = true;
  }

  hideTooltip(): void {
    this.tooltipVisible = false;
  }

  constructor(private readonly socketService: SocketService,
              private readonly system: SystemService) {
    this.socketService.connect();
  }

  ngOnInit(): void {
    this.socketService.on<any>('system_stats', (s) => {
      const cpu = Math.round((s?.cpu?.percent ?? 0));
      const memPct = Math.round((s?.memory?.percent ?? 0));
      const diskPct = Math.round((s?.disk?.percent ?? 0));
      this.cpuLabel = `CPU ${cpu}%`;
      this.memLabel = `RAM ${memPct}%`;
      this.diskLabel = `Disk ${diskPct}%`;
      const gpus = Array.isArray(s?.gpus) ? s.gpus : [];
      if (gpus.length > 0) {
        this.gpuLabels = gpus.map((g: any, i: number) => {
          const name = g?.name ?? `GPU${i}`;
          const loadPct = Math.round(((g?.load ?? g?.memory_util ?? 0) as number) * 100);
          const used = Math.round((g?.memory_used ?? 0));
          const total = Math.round((g?.memory_total ?? 0));
          const toGB = (mb: number) => (mb > 1024 ? (mb / 1024) : mb);
          const isMB = total > 64 && total < 16384;
          const usedOut = isMB ? toGB(used) : used;
          const totalOut = isMB ? toGB(total) : total;
          const unit = isMB ? 'GB' : 'MB';
          return `${name} ${loadPct}% ${usedOut.toFixed(1)}/${totalOut.toFixed(1)}${unit}`;
        });
      } else {
        this.gpuLabels = [];
      }
    });

    this.socketService.on<any>('training_status', (t) => {
      this.trainingFooter = {
        state: t?.state ?? 'idle',
        progress: Math.round(t?.progress ?? 0),
        message: t?.message,
        queue_detect: t?.queue_detect,
        queue_detect_max: t?.queue_detect_max,
        queue_results: t?.queue_results,
        fps: t?.fps
      };
      const st = this.trainingFooter.state;
      if (st !== 'running') {
        this.terminating = false;
        if (st === 'idle' || st === 'cancelled' || st === 'completed' || st === 'failed') {
          this.trainingFooter.message = undefined;
          this.trainingFooter.progress = st === 'idle' ? 0 : this.trainingFooter.progress;
          // Clear thumbnail when not running
          this.trainingThumb = null;
        }
      }
    });

    // Preview thumbnail updates
    this.socketService.on<any>('training_frame', (data) => {
      const img: string | undefined = data?.image;
      if (img && typeof img === 'string') {
        this.trainingThumb = img.startsWith('data:image') ? img : `data:image/jpeg;base64,${img}`;
      }
    });

    // Sync initial state in case training is already running when UI loads
    this.system.getTrainingStatus().subscribe({
      next: (t: any) => {
        this.trainingFooter = {
          state: t?.state ?? 'idle',
          progress: Math.round(t?.progress ?? 0),
          message: t?.message,
          queue_detect: t?.queue_detect,
          queue_detect_max: t?.queue_detect_max,
          queue_results: t?.queue_results,
          fps: t?.fps
        };
        const st = this.trainingFooter.state;
        if (st !== 'running') {
          this.terminating = false;
          if (st === 'idle' || st === 'cancelled' || st === 'completed' || st === 'failed') {
            this.trainingFooter.message = undefined;
            this.trainingFooter.progress = st === 'idle' ? 0 : this.trainingFooter.progress;
          }
        }
      },
      error: () => {}
    });
  }

  terminate(): void {
    if (this.terminating) return;
    this.terminating = true;
    this.system.terminateTraining().subscribe({
      next: () => {},
      error: () => { this.terminating = false; }
    });
  }
}
