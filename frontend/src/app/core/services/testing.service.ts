import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({ providedIn: 'root' })
export class TestingService {
  private readonly baseUrl = '/api/test';

  constructor(private readonly http: HttpClient) {}

  startTest(path: string, threshold: number = 0.7, batchSize: number = 4, useBatching: boolean = true) {
    return this.http.post<{ status: string; batching: boolean; batch_size?: number }>(`${this.baseUrl}`, { 
      path, 
      threshold, 
      batch_size: batchSize, 
      use_batching: useBatching 
    });
  }

  getStatus() {
    return this.http.get<any>(`${this.baseUrl}/status`);
  }

  terminate() {
    return this.http.post<{ status: string }>(`${this.baseUrl}/terminate`, {});
  }
}


