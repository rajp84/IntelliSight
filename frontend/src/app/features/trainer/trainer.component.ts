import { Component, OnDestroy, OnInit, ElementRef, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { SocketService } from '../../core/services/socket.service';

@Component({
  selector: 'app-trainer',
  imports: [CommonModule],
  templateUrl: './trainer.component.html',
  styleUrl: './trainer.component.scss'
})
export class TrainerComponent implements OnInit, OnDestroy {
  frameSrc: string | null = null;
  logs: string[] = [];
  fps: number | null = null;
  qDetect: number | null = null;
  qDetectMax: number | null = null;
  qResults: number | null = null;
  qEmbed: number | null = null;
  qEmbedMax: number | null = null;
  @ViewChild('logContainer') private logContainer?: ElementRef<HTMLDivElement>;

  // Modal state
  showCompletionModal = false;
  trainingStatus: string | null = null;
  trainingId: string | null = null;

  constructor(private readonly socket: SocketService, private readonly router: Router) {}

  ngOnInit(): void {
    this.socket.connect();
    this.socket.on('training_frame', (data: any) => {
      this.frameSrc = data?.image || null;
    });
    this.socket.on('training_log', (data: any) => {
      const msg = (data && data.message) ? String(data.message) : '';
      if (msg) {
        this.logs.push(msg);
        if (this.logs.length > 200) {
          this.logs.splice(0, this.logs.length - 200);
        }
        // Auto-scroll after view updates
        setTimeout(() => {
          const el = this.logContainer?.nativeElement;
          if (el) {
            el.scrollTop = el.scrollHeight;
          }
        }, 0);
      }
    });
    this.socket.on('training_status', (t: any) => {
      this.fps = typeof t?.fps === 'number' ? t.fps : null;
      this.qDetect = typeof t?.queue_detect === 'number' ? t.queue_detect : null;
      this.qDetectMax = typeof t?.queue_detect_max === 'number' ? t.queue_detect_max : null;
      this.qResults = typeof t?.queue_results === 'number' ? t.queue_results : null;
      this.qEmbed = typeof t?.queue_embed === 'number' ? t.queue_embed : null;
      this.qEmbedMax = typeof t?.queue_embed_max === 'number' ? t.queue_embed_max : null;
      
      // Check if training has completed
      if (t?.state === 'completed' || t?.state === 'cancelled' || t?.state === 'error') {
        this.trainingStatus = t.state;
        this.trainingId = t?.training_id || null;
        this.showCompletionModal = true;
      }
    });
  }

  ngOnDestroy(): void {
    // no-op; rely on singleton socket
  }

  closeCompletionModal(): void {
    this.showCompletionModal = false;
    this.trainingStatus = null;
    this.trainingId = null;
  }

  goToResults(): void {
    if (this.trainingId) {
      this.router.navigate(['/results', this.trainingId]);
    }
  }
}
