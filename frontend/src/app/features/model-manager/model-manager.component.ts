import { Component, OnInit, inject } from '@angular/core';
import { CommonModule, NgFor, NgIf } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ModelsService, LocalModel } from '../../core/services/models.service';
import { SocketService } from '../../core/services/socket.service';

@Component({
  selector: 'app-model-manager',
  standalone: true,
  imports: [CommonModule, NgFor, NgIf, FormsModule],
  templateUrl: './model-manager.component.html',
  styleUrl: './model-manager.component.scss'
})
export class ModelManagerComponent implements OnInit {
  private readonly models = inject(ModelsService);
  private readonly socket = inject(SocketService);

  items: LocalModel[] = [];
  defaultModel: string | null = null;
  loading = false;
  // Roboflow
  rfWorkspace = '';
  rfProject = '';
  rfVersion: number | null = null;
  downloading = false;
  // Dataset
  datasetFormat: string = 'yolov8';
  datasetDownloading = false;
  importing = false;
  datasets: Array<{ name: string; size: number; modified: number; path?: string }> = [];
  importStatusText: string | null = null;
  importProgressPct: number | null = null;
  currentImportPath: string | null = null;
  workspaces: Array<{ id: string; name: string; slug: string }> = [];
  projects: Array<{ id: string; name: string; slug: string }> = [];
  versions: Array<{ id: number; name: string; version: number }> = [];

  ngOnInit(): void {
    this.refresh();
    this.refreshDatasets();
    this.loadWorkspaces();
    // Subscribe to dataset import status events
    this.socket.on<any>('dataset_import_status', (s) => {
      try {
        if (!s) return;
        // Match by dataset_path to ensure we update the current import only
        if (this.currentImportPath && s.dataset_path && typeof s.dataset_path === 'string') {
          if (s.dataset_path === this.currentImportPath) {
            const processed = Number(s.processed || 0);
            const total = Number(s.total || 0);
            const pct = total > 0 ? Math.round((processed / total) * 100) : null;
            this.importProgressPct = pct;
            this.importStatusText = s.message || `Processed ${processed}/${total}`;
            // Log to console as requested
            // eslint-disable-next-line no-console
            console.log('[DatasetImport]', this.importStatusText, { processed, total, training_id: s.training_id });
          }
        }
      } catch {
        // ignore
      }
    });
  }

  refresh(): void {
    this.loading = true;
    this.models.list().subscribe({
      next: (res) => {
        const items = (res.items || []).slice().sort((a, b) => (b.modified || 0) - (a.modified || 0));
        this.items = items;
        this.defaultModel = res.default_model || null;
      },
      error: () => {},
      complete: () => { this.loading = false; }
    });
  }

  refreshDatasets(): void {
    this.models.listRoboflowDatasets().subscribe({
      next: (res) => { this.datasets = res.items || []; },
      error: () => { this.datasets = []; }
    });
  }

  setDefault(it: LocalModel): void {
    this.models.setDefault(it.name).subscribe({ next: () => this.refresh(), error: () => {} });
  }

  deleteModel(it: LocalModel): void {
    if (!it?.name) return;
    if (this.defaultModel === it.name) return; // Guard: don't delete default model
    this.models.deleteModel(it.name).subscribe({ next: () => this.refresh(), error: () => {} });
  }

  download(): void {
    if (!this.rfWorkspace || !this.rfProject || !this.rfVersion) return;
    this.downloading = true;
    this.models.downloadRoboflow(this.rfWorkspace, this.rfProject, this.rfVersion).subscribe({
      next: () => this.refresh(),
      error: () => {},
      complete: () => { this.downloading = false; }
    });
  }

  loadWorkspaces(): void {
    this.models.listRoboflowWorkspaces().subscribe({
      next: (res) => { this.workspaces = res.items || []; },
      error: () => { this.workspaces = []; }
    });
  }

  onWorkspaceChange(): void {
    this.rfProject = '';
    this.projects = [];
    this.versions = [];
    if (!this.rfWorkspace) return;
    this.models.listRoboflowProjects(this.rfWorkspace).subscribe({
      next: (res) => { this.projects = res.items || []; },
      error: () => { this.projects = []; }
    });
  }

  onProjectChange(): void {
    this.rfVersion = null;
    this.versions = [];
    if (!this.rfWorkspace || !this.rfProject) return;
    this.models.listRoboflowVersions(this.rfWorkspace, this.rfProject).subscribe({
      next: (res) => { this.versions = res.items || []; },
      error: () => { this.versions = []; }
    });
  }

  downloadDataset(): void {
    if (!this.rfWorkspace || !this.rfProject || !this.rfVersion) return;
    this.datasetDownloading = true;
    this.models.downloadRoboflowDataset(this.rfWorkspace, this.rfProject, this.rfVersion, this.datasetFormat).subscribe({
      next: () => this.refreshDatasets(),
      error: () => {},
      complete: () => { this.datasetDownloading = false; }
    });
  }

  importDataset(d: { name: string; size: number; modified: number; path?: string }): void {
    const datasetPath = (d as any).path || '';
    if (!datasetPath) return;
    this.importing = true;
    this.currentImportPath = datasetPath;
    this.importProgressPct = null;
    this.importStatusText = 'Starting import…';
    this.models.importRoboflowDataset(datasetPath, 'train', this.datasetFormat).subscribe({
      next: (res) => {
        // eslint-disable-next-line no-console
        console.log('[DatasetImport] Completed', res);
        this.importStatusText = 'Import completed';
        this.importProgressPct = 100;
      },
      error: () => {
        this.importStatusText = 'Import failed';
        this.importing = false;
        this.currentImportPath = null;
      },
      complete: () => { this.importing = false; this.currentImportPath = null; this.refreshDatasets(); }
    });
  }

  deleteDataset(d: { name: string; size: number; modified: number; path?: string }): void {
    if (!d?.name) return;
    this.models.deleteRoboflowDataset(d.name).subscribe({
      next: () => this.refreshDatasets(),
      error: () => {}
    });
  }
}


