import { Component, OnInit, inject } from '@angular/core';
import { CommonModule, NgFor, NgIf } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ModelsService, LocalModel } from '../../core/services/models.service';

@Component({
  selector: 'app-model-manager',
  standalone: true,
  imports: [CommonModule, NgFor, NgIf, FormsModule],
  templateUrl: './model-manager.component.html',
  styleUrl: './model-manager.component.scss'
})
export class ModelManagerComponent implements OnInit {
  private readonly models = inject(ModelsService);

  items: LocalModel[] = [];
  defaultModel: string | null = null;
  loading = false;
  // Roboflow
  rfWorkspace = '';
  rfProject = '';
  rfVersion: number | null = null;
  downloading = false;
  workspaces: Array<{ id: string; name: string; slug: string }> = [];
  projects: Array<{ id: string; name: string; slug: string }> = [];
  versions: Array<{ id: number; name: string; version: number }> = [];

  ngOnInit(): void {
    this.refresh();
    this.loadWorkspaces();
  }

  refresh(): void {
    this.loading = true;
    this.models.list().subscribe({
      next: (res) => { this.items = res.items || []; this.defaultModel = res.default_model || null; },
      error: () => {},
      complete: () => { this.loading = false; }
    });
  }

  setDefault(it: LocalModel): void {
    this.models.setDefault(it.name).subscribe({ next: () => this.refresh(), error: () => {} });
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
}


