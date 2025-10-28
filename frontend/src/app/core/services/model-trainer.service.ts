import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({ providedIn: 'root' })
export class ModelTrainerService {
  private readonly baseUrl = '/api/train-model';

  constructor(private readonly http: HttpClient) {}

  start(epochs: number = 100, yoloModel: string = 'yolov8n.pt', synthPerImage?: number) {
    const body: any = { epochs, yolo_model: yoloModel };
    if (synthPerImage !== undefined && synthPerImage !== null) body.synth_per_image = synthPerImage;
    return this.http.post<{ status: string }>(`${this.baseUrl}`, body);
  }

  status() {
    return this.http.get<any>(`${this.baseUrl}/status`);
  }

  terminate() {
    return this.http.post<{ status: string }>(`${this.baseUrl}/terminate`, {});
  }
}


