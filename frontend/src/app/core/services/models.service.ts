import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface LocalModel { name: string; size: number; modified: number }

@Injectable({ providedIn: 'root' })
export class ModelsService {
  private readonly http = inject(HttpClient);

  list(): Observable<{ items: LocalModel[]; default_model?: string }>{
    return this.http.get<{ items: LocalModel[]; default_model?: string }>(`/api/models`);
  }

  setDefault(name: string): Observable<{ status: string; default_model: string }>{
    return this.http.post<{ status: string; default_model: string }>(`/api/models/default`, { name });
  }

  setRoboflowKey(api_key: string): Observable<{ status: string }>{
    return this.http.post<{ status: string }>(`/api/models/roboflow/key`, { api_key });
  }

  downloadRoboflow(workspace: string, project: string, version: number, format: string = 'yolov8')
    : Observable<{ status: string; files: string[] }>{
    return this.http.post<{ status: string; files: string[] }>(`/api/models/roboflow/download`, {
      workspace, project, version, format
    });
  }

  listRoboflowWorkspaces(): Observable<{ items: Array<{ id: string; name: string; slug: string }> }>{
    return this.http.get<{ items: Array<{ id: string; name: string; slug: string }> }>(`/api/models/roboflow/workspaces`);
  }

  listRoboflowProjects(workspace: string): Observable<{ items: Array<{ id: string; name: string; slug: string }> }>{
    return this.http.get<{ items: Array<{ id: string; name: string; slug: string }> }>(`/api/models/roboflow/projects`, { params: { workspace } });
  }

  listRoboflowVersions(workspace: string, project: string): Observable<{ items: Array<{ id: number; name: string; version: number }> }>{
    return this.http.get<{ items: Array<{ id: number; name: string; version: number }> }>(`/api/models/roboflow/versions`, { params: { workspace, project } });
  }
}


