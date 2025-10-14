import { Component, OnInit } from '@angular/core';
import { NgIf, NgFor } from '@angular/common';
import { RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';
import { SocketService } from './core/services/socket.service';
import { SystemService } from './core/services/system.service';
import { TrainingService } from './core/services/training.service';
import { ModelTrainerService } from './core/services/model-trainer.service';
import { SplashComponent } from './features/splash/splash.component';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, RouterLink, RouterLinkActive, NgIf, NgFor, SplashComponent],
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
  mobileMenuOpen = false;

  cpuLabel = '';
  memLabel = '';
  diskLabel = '';
  gpuLabels: string[] = [];
  trainingThumb: string | null = null;
  trainingFooter: { state: string; progress: number; message?: string; queue_detect?: number; queue_detect_max?: number; queue_results?: number; queue_embed?: number; queue_embed_max?: number; fps?: number } | null = null;
  terminating = false;
  isModelTrainerRunning = false;

  showTooltip(event: MouseEvent, text: string): void {
    const target = event.currentTarget as HTMLElement;
    const rect = target.getBoundingClientRect();
    // Position relative to viewport, not sidebar
    this.tooltipTop = rect.top + rect.height / 2;
    this.tooltipLeft = rect.right + 8; // account for arrow overhang so tip touches icon
    this.tooltipText = text;
    this.tooltipVisible = true;
  }

  hideTooltip(): void {
    this.tooltipVisible = false;
  }

  toggleMobileMenu(): void {
    this.mobileMenuOpen = !this.mobileMenuOpen;
  }

  closeMobileMenu(): void {
    this.mobileMenuOpen = false;
  }

  constructor(private readonly socketService: SocketService,
              private readonly system: SystemService,
              private readonly training: TrainingService,
              private readonly modelTrainer: ModelTrainerService) {
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
        queue_embed: t?.queue_embed,
        queue_embed_max: t?.queue_embed_max,
        fps: t?.fps
      };
      const st = this.trainingFooter.state;
      if (st !== 'running') {
        this.terminating = false;
        if (st === 'idle' || st === 'cancelled' || st === 'completed' || st === 'failed') {
          if (this.trainingFooter) {
            this.trainingFooter.message = undefined;
            this.trainingFooter.progress = st === 'idle' ? 0 : this.trainingFooter.progress;
          }
          // Clear thumbnail when not running
          this.trainingThumb = null;
        }
      } else {
        // Classic trainer running (not model trainer)
        this.isModelTrainerRunning = false;
      }
    });

    // Also listen to model trainer status (primary event from trainer)
    this.socketService.on<any>('model_trainer_status', (t) => {
      this.trainingFooter = {
        state: t?.state ?? 'idle',
        progress: Math.round(t?.progress ?? 0),
        message: t?.message,
        queue_detect: t?.queue_detect,
        queue_detect_max: t?.queue_detect_max,
        queue_results: t?.queue_results,
        queue_embed: t?.queue_embed,
        queue_embed_max: t?.queue_embed_max,
        fps: t?.fps
      } as any;
      const st = this.trainingFooter?.state;
      if (st && st !== 'running') {
        this.terminating = false;
        if (st === 'idle' || st === 'cancelled' || st === 'completed' || st === 'failed') {
          if (this.trainingFooter) {
            this.trainingFooter.message = undefined;
            this.trainingFooter.progress = st === 'idle' ? 0 : this.trainingFooter.progress;
          }
          this.trainingThumb = null;
        }
        this.isModelTrainerRunning = false;
      } else if (st === 'running') {
        this.isModelTrainerRunning = true;
      }
    });

    // Keep the footer message updated with the last training log line
    this.socketService.on<any>('model_trainer_log', (l) => {
      const msg = typeof l?.message === 'string' ? l.message : undefined;
      if (!msg) return;
      if (!this.trainingFooter) {
        this.trainingFooter = { state: 'running', progress: 0, message: msg } as any;
      } else {
        this.trainingFooter.message = msg;
      }
    });

    // Preview thumbnail updates (training)
    this.socketService.on<any>('training_frame', (data) => {
      const img: string | undefined = data?.image;
      if (img && typeof img === 'string') {
        this.trainingThumb = img.startsWith('data:image') ? img : `data:image/jpeg;base64,${img}`;
      }
    });

    // Sync initial state in case training is already running when UI loads
    this.training.getTrainingStatus().subscribe({
      next: (t: any) => {
        this.trainingFooter = {
          state: t?.state ?? 'idle',
          progress: Math.round(t?.progress ?? 0),
          message: t?.message,
          queue_detect: t?.queue_detect,
          queue_detect_max: t?.queue_detect_max,
          queue_results: t?.queue_results,
          queue_embed: t?.queue_embed,
          queue_embed_max: t?.queue_embed_max,
          fps: t?.fps
        };
        const st = this.trainingFooter.state;
        if (st !== 'running') {
          this.terminating = false;
          if (st === 'idle' || st === 'cancelled' || st === 'completed' || st === 'failed') {
            if (this.trainingFooter) {
              this.trainingFooter.message = undefined;
              this.trainingFooter.progress = st === 'idle' ? 0 : this.trainingFooter.progress;
            }
          }
        }
      },
      error: () => {}
    });

    // Sync initial model-trainer state
    this.modelTrainer.status().subscribe({
      next: (t: any) => {
        const st = t?.state ?? 'idle';
        if (st === 'running' || st === 'terminating') {
          this.trainingFooter = {
            state: st,
            progress: Math.round(t?.progress ?? 0),
            message: t?.message,
            queue_detect: t?.queue_detect,
            queue_detect_max: t?.queue_detect_max,
            queue_results: t?.queue_results,
            queue_embed: t?.queue_embed,
            queue_embed_max: t?.queue_embed_max,
            fps: t?.fps
          } as any;
          this.isModelTrainerRunning = (st === 'running' || st === 'terminating');
        }
      },
      error: () => {}
    });

  }

  terminate(): void {
    if (this.terminating) return;
    this.terminating = true;
    this.training.terminateTraining().subscribe({
      next: () => {},
      error: () => { this.terminating = false; }
    });
  }

  
}
