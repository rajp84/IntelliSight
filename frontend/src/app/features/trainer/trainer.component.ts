import { Component, OnDestroy, OnInit, ElementRef, ViewChild, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { SocketService } from '../../core/services/socket.service';
import { HttpClient } from '@angular/common/http';

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

  // Auto discovery state
  discoveredPhrases: string = '';
  lastCaption: string = '';
  discoveryFrameSrc: string = '';
  showAutoDiscovery: boolean = false;

  constructor(
    private readonly socket: SocketService, 
    private readonly router: Router, 
    private readonly http: HttpClient,
    private readonly cdr: ChangeDetectorRef
  ) {}

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
    
    // Listen for training caption events
    this.socket.on('training_caption', (data: any) => {
      // Parse the special message format: "TRAINING_CAPTION:caption|phrases|frame|base64Image"
      if (data?.message && data.message.startsWith('TRAINING_CAPTION:')) {
        const parts = data.message.substring(17).split('|'); // Remove "TRAINING_CAPTION:" prefix
        if (parts.length >= 3) {
          const caption = parts[0];
          const phrases = parts[1];
          const frame = parts[2];
          const base64Image = parts[3] || '';
          
          this.lastCaption = caption;
          this.discoveredPhrases = phrases;
          this.showAutoDiscovery = true;
          
          // Set the discovery frame image if available
          if (base64Image) {
            this.discoveryFrameSrc = `data:image/jpeg;base64,${base64Image}`;
          }
          
          this.cdr.detectChanges();
        }
      }
      // Handle old format for backward compatibility
      else if (data?.phrases) {
        this.discoveredPhrases = data.phrases;
        this.showAutoDiscovery = true;
        this.cdr.detectChanges();
      }
      if (data?.caption) {
        this.lastCaption = data.caption;
        this.showAutoDiscovery = true;
        this.cdr.detectChanges();
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
