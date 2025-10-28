import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, Observable, tap } from 'rxjs';

export interface TrainingParams {
  batch_size?: number;
  detect_every?: number;
  frame_stride?: number;
  max_new_tokens?: number;
  mosaic_cols?: number;
  mosaic_tile_scale?: number;
  resize_width?: number;
  detection_debug?: boolean;
  tf32?: boolean;
  score_threshold?: number;
  interpolate_boxes?: boolean;
  dtype_mode?: 'auto' | 'float16' | 'float32' | 'bfloat16';
  embedding_batch_size?: number;
}

export interface SystemConfig {
  library_path: string;
  training_params: TrainingParams;
  library_file_extensions: string;
  hf_token?: string | null;
  roboflow_api_key?: string | null;
  florence_model: string;
  dinov3_model: string;
  dinov3_dimension: number;
}

@Injectable({
  providedIn: 'root'
})
export class SystemService {
  private readonly baseUrl = '/api/system';
  private readonly configSubject = new BehaviorSubject<SystemConfig | null>(null);
  readonly config$ = this.configSubject.asObservable();

  constructor(private readonly http: HttpClient) { }

  getConfig<T = SystemConfig>(): Observable<T> {
    return this.http.get<T>(`${this.baseUrl}/config`).pipe(
      tap((cfg) => this.configSubject.next(cfg as unknown as SystemConfig))
    );
  }

  saveConfig<T = SystemConfig>(payload: T): Observable<T> {
    return this.http.post<T>(`${this.baseUrl}/config`, payload).pipe(
      tap((cfg) => this.configSubject.next(cfg as unknown as SystemConfig))
    );
  }

  getConfigSnapshot(): SystemConfig | null {
    return this.configSubject.value;
  }

  setConfigLocal(cfg: SystemConfig): void {
    this.configSubject.next(cfg);
  }

  // ------- Library APIs -------
  listLibrary(path: string = ''): Observable<{ root: string; path: string; items: Array<{ name: string; type: 'file' | 'dir' }> }> {
    const params = path ? { params: { path } } : {};
    return this.http.get<{ root: string; path: string; items: Array<{ name: string; type: 'file' | 'dir' }> }>(`${this.baseUrl}/library/list`, params as unknown as undefined);
  }

  getLibraryFileUrl(path: string): string {
    const url = new URL(`${this.baseUrl}/library/file`, window.location.origin);
    url.searchParams.set('path', path);
    return url.toString();
  }

  // ------- Models -------
  listModels() {
    return this.http.get<{ items: Array<{ name: string; path: string }> }>(`${this.baseUrl}/models`);
  }

  uploadLibraryFile(file: File, path: string = ''): Observable<{ status: string; filename: string; path: string; size: number }> {
    const formData = new FormData();
    formData.append('file', file);
    
    const params = path ? { params: { path } } : {};
    return this.http.post<{ status: string; filename: string; path: string; size: number }>(
      `${this.baseUrl}/library/upload`, 
      formData, 
      { ...params, reportProgress: false }
    );
  }

  createLibraryFolder(name: string, path: string = ''): Observable<{ status: string; message: string; path: string }> {
    const params = { 
      name,
      ...(path ? { path } : {})
    };
    return this.http.post<{ status: string; message: string; path: string }>(
      `${this.baseUrl}/library/folder`, 
      null,
      { params }
    );
  }

  // Training APIs were moved to TrainingService
}
