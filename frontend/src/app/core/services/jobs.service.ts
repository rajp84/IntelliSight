import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface JobItem {
  job_id: string;
  media_file: string;
  status: 'accepted' | 'queued' | 'running' | 'completed' | 'failed';
  type?: string;
  conf?: number;
  progress?: number;
  step?: number;
  total_steps?: number;
  preview?: string;
  model_path?: string;
  model_name?: string;
  frames_processed?: number;
  total_frames?: number;
  created_at?: string;
  started_at?: string;
  completed_at?: string;
  failed_at?: string;
  error?: string;
}

@Injectable({ providedIn: 'root' })
export class JobsService {
  private readonly baseUrl = '/api/process';

  constructor(private readonly http: HttpClient) {}

  listJobs(limit: number = 200): Observable<{ items: JobItem[] }> {
    return this.http.get<{ items: JobItem[] }>(`${this.baseUrl}/jobs`, { params: { limit } as any });
  }

  getJobFrames(jobId: string, afterIdx: number = -1, blocks: number = 3): Observable<{ frames: any[]; last_idx: number; imgw?: number; imgh?: number }> {
    return this.http.get<{ frames: any[]; last_idx: number; imgw?: number; imgh?: number }>(`${this.baseUrl}/jobs/${encodeURIComponent(jobId)}/frames`, {
      params: { after_idx: afterIdx, blocks } as any,
    });
  }

  getJob(jobId: string): Observable<any> {
    return this.http.get<any>(`${this.baseUrl}/jobs/${encodeURIComponent(jobId)}`);
  }

  deleteJob(jobId: string): Observable<{ deleted: boolean; frame_buckets_deleted?: number }> {
    return this.http.delete<{ deleted: boolean; frame_buckets_deleted?: number }>(`${this.baseUrl}/jobs/${encodeURIComponent(jobId)}`);
  }
}


