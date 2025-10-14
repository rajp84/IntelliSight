import { Component, ElementRef, ViewChild, AfterViewInit, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { ModelTrainerService } from '../../core/services/model-trainer.service';
import { SocketService } from '../../core/services/socket.service';
import { SystemService } from '../../core/services/system.service';

@Component({
  selector: 'app-model-trainer',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './model-trainer.component.html',
  styleUrl: './model-trainer.component.scss'
})
export class ModelTrainerComponent implements AfterViewInit, OnInit {
  status: any = null;
  logs: string[] = [];
  tableRows: Array<string[]> = [];
  @ViewChild('logContainer') logContainer!: ElementRef<HTMLDivElement>;
  isStarting = false;
  epochs: number = 100;
  synthPerImage: number = 200;
  best: { epoch: number; precision?: number; recall?: number; map50?: number; map5095?: number } | null = null;
  models: Array<{ name: string; path: string }> = [];
  startingWeights: string = 'yolov8n.pt';

  constructor(private readonly svc: ModelTrainerService, private readonly socket: SocketService, private readonly system: SystemService) {
    this.socket.on<any>('model_trainer_status', (s) => {
      this.status = s;
      // Keep Start disabled while running/terminating; re-enable otherwise
      const st = this.status?.state;
      this.isStarting = (st === 'running' || st === 'terminating');
    });
    this.socket.on<any>('model_trainer_log', (l) => {
      const raw = l?.message as string | undefined;
      if (raw === undefined || raw === null) return;
      // Only map truly empty string to a single space; keep other content unchanged
      const m = raw.length === 0 ? ' ' : raw;
      // Attempt to parse Ultralytics table lines into columns for nicer rendering
      const clean = m.replace(/\u2500|\u2501|\u2502|\u2503|\u2514|\u2518|\u2510|\u250C|\u252C|\u2534|\u253C|\u251C|\u2524|\u25A0|\u25A1|\u2591|\u2592|\u2593/g, '').trim();
      if (/^Class\s+Images\s+Instances/i.test(clean) || /^\s*all\s+\d+\s+\d+/.test(clean)) {
        // Split by multiple spaces
        const cols = clean.split(/\s{2,}/).filter(Boolean);
        if (cols.length >= 4) {
          this.tableRows.push(cols);
          if (this.tableRows.length > 200) this.tableRows.shift();
          // Track best row when 'all' metrics appear
          // Expected columns: [Class, Images, Instances, Box(P), R, mAP50, mAP50-95]
          try {
            const isAll = /^all$/i.test(cols[0]);
            if (isAll && cols.length >= 7) {
              const precision = parseFloat(cols[3]);
              const recall = parseFloat(cols[4]);
              const map50 = parseFloat(cols[5]);
              const map5095 = parseFloat(cols[6]);
              const curEpoch = (this.status?.epoch as number) || 0;
              const better = (!this.best || (isFinite(map5095) && (this.best.map5095 ?? -1) < map5095));
              if (better) {
                this.best = { epoch: curEpoch, precision, recall, map50, map5095 };
              }
            }
          } catch {}
          return; // don't also push to raw logs
        }
      }
      this.logs.push(m);
      if (this.logs.length > 500) this.logs.shift();
      this.scrollLogsToBottom();
    });
  }

  ngOnInit(): void {
    // Hydrate from last known backend status so UI shows immediately on navigation
    try {
      this.svc.status().subscribe({
        next: (s) => {
          this.status = s;
          if (s?.best && typeof s.best === 'object') {
            const b = s.best as any;
            const epoch = Number(b.epoch || 0);
            const precision = isFinite(Number(b.precision)) ? Number(b.precision) : undefined;
            const recall = isFinite(Number(b.recall)) ? Number(b.recall) : undefined;
            const map50 = isFinite(Number(b.map50)) ? Number(b.map50) : undefined;
            const map5095 = isFinite(Number(b.map5095)) ? Number(b.map5095) : undefined;
            this.best = { epoch, precision, recall, map50, map5095 } as any;
          }
          const st = this.status?.state;
          this.isStarting = (st === 'running' || st === 'terminating');
        },
        error: () => {}
      });
    } catch {}
    // Load available models from server; include built-ins
    try {
      this.system.listModels().subscribe({
        next: (res) => {
          const items = res.items || [];
          // Prepend common Ultralytics weights
          const builtins: Array<{ name: string; path: string }> = [
            { name: 'YOLOv8 Nano (yolov8n.pt)', path: 'yolov8n.pt' },
            { name: 'YOLOv8 Small (yolov8s.pt)', path: 'yolov8s.pt' },
            { name: 'YOLOv8 Medium (yolov8m.pt)', path: 'yolov8m.pt' },
            { name: 'YOLOv8 Large (yolov8l.pt)', path: 'yolov8l.pt' },
          ];
          this.models = [...builtins, ...items];
          // Default to first if none selected
          if (!this.startingWeights && this.models.length > 0) {
            this.startingWeights = this.models[0].path;
          }
        },
        error: () => {}
      });
    } catch {}
  }

  ngAfterViewInit(): void {
    this.scrollLogsToBottom();
  }

  private scrollLogsToBottom(): void {
    try {
      const el = this.logContainer?.nativeElement;
      if (!el) return;
      el.scrollTop = el.scrollHeight;
    } catch {}
  }

  start(): void {
    if (this.isStarting) return;
    if (!this.epochs || this.epochs < 1) { this.epochs = 100; }
    this.isStarting = true;
    const weights = this.startingWeights || 'yolov8n.pt';
    this.svc.start(this.epochs, weights, this.synthPerImage).subscribe({
      next: () => {},
      error: () => { this.isStarting = false; }
    });
  }
  terminate(): void { this.svc.terminate().subscribe(); }
}
