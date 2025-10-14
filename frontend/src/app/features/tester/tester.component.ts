import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SocketService } from '../../core/services/socket.service';
import { TestingService } from '../../core/services/testing.service';

@Component({
  selector: 'app-tester',
  imports: [CommonModule],
  templateUrl: './tester.component.html',
  styleUrl: './tester.component.scss'
})
export class TesterComponent implements OnInit, OnDestroy {
  frameSrc: string | null = null;
  matches: Array<{ id: string; similarity: number; distance: number; payload: any }> = [];
  status: any = null;
  terminating = false;

  constructor(private readonly socket: SocketService, private readonly testing: TestingService) {}

  ngOnInit(): void {
    this.socket.connect();
    this.socket.on<any>('testing_frame', (data) => {
      const img: string | undefined = data?.image;
      if (img && typeof img === 'string') {
        this.frameSrc = img.startsWith('data:image') ? img : `data:image/jpeg;base64,${img}`;
      }
      // If matches included with the frame, render them synchronously
      const m = Array.isArray(data?.matches) ? data.matches : [];
      if (m.length) {
        this.matches = m.map((x: any) => ({ id: String(x.id), similarity: Number(x.similarity ?? 0), distance: Number(x.distance ?? 0), payload: x.payload }));
      }
    });
    this.socket.on<any>('testing_matches', (data) => {
      // If backend included the frame in this event, render it to keep frame/matches in sync
      const imgInMatches: string | undefined = data?.image;
      if (imgInMatches && typeof imgInMatches === 'string') {
        this.frameSrc = imgInMatches.startsWith('data:image') ? imgInMatches : `data:image/jpeg;base64,${imgInMatches}`;
      }
      const m = Array.isArray(data?.matches) ? data.matches : [];
      console.log(`[TESTER] Received matches:`, m);
      this.matches = m.map((x: any) => {
        console.log(`[TESTER] Processing match:`, x);
        return { id: String(x.id), similarity: Number(x.similarity ?? 0), distance: Number(x.distance ?? 0), payload: x.payload };
      });
    });
    this.socket.on<any>('testing_status', (t) => {
      this.status = t;
      if (t?.state && t.state !== 'running') {
        this.terminating = false;
      }
    });
    // Sync initial status
    this.testing.getStatus().subscribe({ next: (t) => (this.status = t), error: () => {} });
  }

  ngOnDestroy(): void {}

  terminate(): void {
    if (this.terminating) return;
    this.terminating = true;
    this.testing.terminate().subscribe({ next: () => {}, error: () => (this.terminating = false) });
  }

  getThumbnailUrl(imageId: string): string {
    // Use relative URL
    const url = `/api/train/things/image/${imageId}.jpg`;
    console.log(`[TESTER] Getting thumbnail for imageId: ${imageId}, URL: ${url}`);
    return url;
  }

  onImageError(event: any): void {
    console.error(`[TESTER] Image load error:`, event);
    // Hide the image and show placeholder
    const img = event.target as HTMLImageElement;
    img.style.display = 'none';
    const placeholder = img.nextElementSibling as HTMLElement;
    if (placeholder) {
      placeholder.style.display = 'flex';
    }
  }
}
