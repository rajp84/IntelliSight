import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { SystemService } from '../../core/services/system.service';
import { TrainingService } from '../../core/services/training.service';
import { SocketService } from '../../core/services/socket.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-library',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './library.component.html',
  styleUrl: './library.component.scss',
  styles: [`
    .drag-over {
      background-color: rgba(59, 130, 246, 0.05);
    }
    
    .drag-over ul {
      opacity: 0.5;
    }
  `]
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
  discoveryInterval: number = 90;
  showGroundingModal = false;
  groundingPhrase = '';
  
  // Upload functionality
  isUploading = false;
  uploadProgress = 0;
  showUploadModal = false;
  selectedFile: File | null = null;
  
  // Create folder functionality
  showCreateFolderModal = false;
  newFolderName = '';
  isCreatingFolder = false;
  
  // Drag and drop functionality
  isDragOver = false;
  dragCounter = 0;

  constructor(private readonly system: SystemService,
              private readonly training: TrainingService,
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
    this.training.startTraining({ 
      path: this.selectedFileRelPath, 
      mosaic: true,
      task_type: '<OD>'
    }).subscribe({
      next: () => {
        this.trainingState = 'running';
        this.router.navigateByUrl('/trainer');
      },
      error: (err: unknown) => {
        this.trainingState = 'failed';
        console.error(err);
      }
    });
  }

  openGroundingModal(): void {
    this.showGroundingModal = true;
    this.groundingPhrase = '';
  }

  closeGroundingModal(): void {
    this.showGroundingModal = false;
    this.groundingPhrase = '';
  }

  startGroundingTraining(): void {
    if (!this.selectedFileRelPath || !this.groundingPhrase.trim()) return;
    
    this.training.startTraining({ 
      path: this.selectedFileRelPath, 
      mosaic: true,
      task_type: '<CAPTION_TO_PHRASE_GROUNDING>',
      phrase: this.groundingPhrase.trim()
    }).subscribe({
      next: () => {
        this.trainingState = 'running';
        this.closeGroundingModal();
        this.router.navigateByUrl('/trainer');
      },
      error: (err: unknown) => {
        this.trainingState = 'failed';
        console.error(err);
      }
    });
  }

  startAutoGroundingTraining(): void {
    if (!this.selectedFileRelPath) return;
    
    this.training.startTraining({ 
      path: this.selectedFileRelPath, 
      mosaic: true,
      task_type: '<CAPTION_TO_PHRASE_GROUNDING>',
      auto_discovery: true,
      discovery_interval: this.discoveryInterval
    }).subscribe({
      next: () => {
        this.trainingState = 'running';
        this.router.navigateByUrl('/trainer');
      },
      error: (err: unknown) => {
        this.trainingState = 'failed';
        console.error(err);
      }
    });
  }

  startDenseCaptionTraining(): void {
    if (!this.selectedFileRelPath) return;
    
    this.training.startTraining({ 
      path: this.selectedFileRelPath, 
      mosaic: true,
      task_type: '<DENSE_REGION_CAPTION>'
    }).subscribe({
      next: () => {
        this.trainingState = 'running';
        this.router.navigateByUrl('/trainer');
      },
      error: (err: unknown) => {
        this.trainingState = 'failed';
        console.error(err);
      }
    });
  }

  // Upload functionality
  openUploadModal(): void {
    this.showUploadModal = true;
    this.selectedFile = null;
    this.uploadProgress = 0;
  }

  closeUploadModal(): void {
    this.showUploadModal = false;
    this.selectedFile = null;
    this.uploadProgress = 0;
    this.isUploading = false;
  }

  // Create folder methods
  openCreateFolderModal(): void {
    this.showCreateFolderModal = true;
    this.newFolderName = '';
    this.isCreatingFolder = false;
  }

  closeCreateFolderModal(): void {
    this.showCreateFolderModal = false;
    this.newFolderName = '';
    this.isCreatingFolder = false;
  }

  createFolder(): void {
    if (!this.newFolderName.trim()) {
      return;
    }

    this.isCreatingFolder = true;
    
    this.system.createLibraryFolder(this.newFolderName.trim(), this.currentPath).subscribe({
      next: (response) => {
        console.log('Folder created successfully:', response);
        // Reload the current directory to show the new folder
        this.load(this.currentPath);
        this.closeCreateFolderModal();
      },
      error: (err) => {
        console.error('Failed to create folder:', err);
        this.error = err.error?.detail || 'Failed to create folder';
        this.isCreatingFolder = false;
      }
    });
  }

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.selectedFile = input.files[0];
    }
  }

  uploadFile(): void {
    if (!this.selectedFile) return;
    
    this.isUploading = true;
    this.uploadProgress = 0;
    
    this.system.uploadLibraryFile(this.selectedFile, this.currentPath).subscribe({
      next: (response) => {
        console.log('Upload successful:', response);
        this.uploadProgress = 100;
        // Reload the current directory to show the new file
        this.load(this.currentPath);
        this.closeUploadModal();
      },
      error: (err) => {
        console.error('Upload failed:', err);
        this.isUploading = false;
        this.uploadProgress = 0;
        // You might want to show an error message to the user
      }
    });
  }

  // Drag and drop functionality
  onDragEnter(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.dragCounter++;
    if (event.dataTransfer?.items && event.dataTransfer.items.length > 0) {
      this.isDragOver = true;
    }
  }

  onDragLeave(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.dragCounter--;
    if (this.dragCounter === 0) {
      this.isDragOver = false;
    }
  }

  onDragOver(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDragOver = false;
    this.dragCounter = 0;

    if (event.dataTransfer?.files && event.dataTransfer.files.length > 0) {
      const files = Array.from(event.dataTransfer.files);
      
      // For now, only handle the first file
      if (files.length > 0) {
        this.uploadFileFromDrop(files[0]);
      }
    }
  }

  private uploadFileFromDrop(file: File): void {
    this.selectedFile = file;
    this.isUploading = true;
    this.uploadProgress = 0;
    
    this.system.uploadLibraryFile(file, this.currentPath).subscribe({
      next: (response) => {
        console.log('Upload successful:', response);
        this.uploadProgress = 100;
        // Reload the current directory to show the new file
        this.load(this.currentPath);
        this.selectedFile = null;
        this.isUploading = false;
        this.uploadProgress = 0;
      },
      error: (err) => {
        console.error('Upload failed:', err);
        this.isUploading = false;
        this.uploadProgress = 0;
        this.selectedFile = null;
      }
    });
  }
}
