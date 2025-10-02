import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({ providedIn: 'root' })
export class TrainingService {
	private readonly trainUrl = '/api/train';

	constructor(private readonly http: HttpClient) {}

	startTraining(payload: { path: string; mosaic?: boolean; extra_args?: any; task_type?: string; phrase?: string; auto_discovery?: boolean; discovery_interval?: number }) {
		return this.http.post<{ status: string }>(`${this.trainUrl}`, payload);
	}

	getTrainingStatus() {
		return this.http.get<any>(`${this.trainUrl}/status`);
	}

	terminateTraining() {
		return this.http.post<any>(`${this.trainUrl}/terminate`, {});
	}

	reconcileRunning() {
		return this.http.post<{ updated: number }>(`${this.trainUrl}/reconcile`, {});
	}

	listTrainingRuns(page: number = 1, pageSize: number = 20) {
		return this.http.get<{ total: number; items: Array<{ _id: string; status: string; file_name: string; started_at: number; completed_at: number | null; frames_processed?: number; total_frames?: number; elapsed_s?: number }> }>(
			`${this.trainUrl}/runs`,
			{ params: { page, page_size: pageSize } as any }
		);
	}

	deleteTrainingRun(runId: string) {
		return this.http.delete<{ deleted: boolean; cleanup_results?: any }>(`${this.trainUrl}/runs/${encodeURIComponent(runId)}`);
	}
}
