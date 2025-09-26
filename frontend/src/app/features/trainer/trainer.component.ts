import { Component, OnDestroy, OnInit, ElementRef, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
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
  @ViewChild('logContainer') private logContainer?: ElementRef<HTMLDivElement>;

  constructor(private readonly socket: SocketService) {}

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
    });
  }

  ngOnDestroy(): void {
    // no-op; rely on singleton socket
  }
}
