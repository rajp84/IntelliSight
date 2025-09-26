import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SystemService } from '../../core/services/system.service';
import { SocketService } from '../../core/services/socket.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-library',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './library.component.html',
  styleUrl: './library.component.scss'
})
export class LibraryComponent {
  entries: Array<{ name: string; type: 'file' | 'dir' }> = [];
  currentPath = '';
  loading = false;
  error = '';
  previewUrl: string | null = null;
  previewType: 'image' | 'video' | 'other' | null = null;
  selectedFileRelPath: string | null = null;
  trainingState: 'idle' | 'running' | 'completed' | 'failed' = 'idle';
  trainingProgress = 0;

  constructor(private readonly system: SystemService,
              private readonly socket: SocketService,
              private readonly router: Router) {}

  ngOnInit(): void {
    this.load();
    this.socket.connect();
    this.socket.on('training_status', (status: any) => {
      this.trainingState = status.state;
      this.trainingProgress = status.progress ?? 0;
    });
  }

  load(path: string = ''): void {
    this.loading = true;
    this.error = '';
    this.system.listLibrary(path).subscribe({
      next: (res) => {
        this.entries = res.items;
        this.currentPath = res.path;
        this.loading = false;
      },
      error: (err) => {
        this.error = 'Failed to load library';
        this.loading = false;
        console.error(err);
      }
    });
  }

  open(entry: { name: string; type: 'file' | 'dir' }): void {
    if (entry.type === 'dir') {
      const next = this.currentPath ? this.currentPath + '/' + entry.name : entry.name;
      this.load(next);
      this.clearPreview();
    } else {
      this.select(entry);
    }
  }

  up(): void {
    if (!this.currentPath) return;
    const parts = this.currentPath.split('/').filter(Boolean);
    parts.pop();
    const upPath = parts.join('/');
    this.load(upPath);
  }

  fileUrl(entry: { name: string; type: 'file' | 'dir' }): string | null {
    if (entry.type !== 'file') return null;
    const p = this.currentPath ? this.currentPath + '/' + entry.name : entry.name;
    return this.system.getLibraryFileUrl(p);
  }

  select(entry: { name: string; type: 'file' | 'dir' }): void {
    if (entry.type !== 'file') return;
    const url = this.fileUrl(entry);
    if (!url) return;
    this.selectedFileRelPath = this.currentPath ? this.currentPath + '/' + entry.name : entry.name;
    const ext = entry.name.split('.').pop()?.toLowerCase() ?? '';
    const imageExt = ['png', 'jpg', 'jpeg', 'gif', 'webp', 'svg'];
    const videoExt = ['mp4', 'webm', 'ogg'];
    if (imageExt.includes(ext)) {
      this.previewType = 'image';
    } else if (videoExt.includes(ext)) {
      this.previewType = 'video';
    } else {
      this.previewType = 'other';
    }
    this.previewUrl = url;
  }

  clearPreview(): void {
    this.previewUrl = null;
    this.previewType = null;
    this.selectedFileRelPath = null;
  }

  startTraining(): void {
    if (!this.selectedFileRelPath) return;
    this.system.startTraining({ path: this.selectedFileRelPath, mosaic: true }).subscribe({
      next: () => {
        this.trainingState = 'running';
        this.router.navigateByUrl('/trainer');
      },
      error: (err) => {
        this.trainingState = 'failed';
        console.error(err);
      }
    });
  }
}
