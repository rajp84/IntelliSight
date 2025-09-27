import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface EmbeddingItem {
  id: string;
  payload: {
    label?: string;
    score?: number;
    bbox?: [number, number, number, number];
    frame_index?: number;
    track_id?: number | null;
    unique_id?: string;
  };
}

export interface EmbeddingsResponse {
  items: EmbeddingItem[];
}

@Injectable({
  providedIn: 'root'
})
export class ResultService {
  private readonly http = inject(HttpClient);

  /**
   * Get embeddings for a specific training run
   */
  getEmbeddings(runId: string, limit: number = 10000, offset: number = 0): Observable<EmbeddingsResponse> {
    return this.http.get<EmbeddingsResponse>(`/api/train/runs/${encodeURIComponent(runId)}/embeddings?limit=${limit}&offset=${offset}`);
  }

  /**
   * Get a training run image by ID
   */
  getImageUrl(runId: string, milvusId: string, imageId?: string): string {
    // Use image_id from payload if available, otherwise fall back to milvusId
    const filename = imageId ? `${imageId}.jpg` : `${runId}_${milvusId}.jpg`;
    return `/api/train/runs/${encodeURIComponent(runId)}/image/${encodeURIComponent(filename)}`;
  }

  /**
   * Update labels for multiple embeddings
   */
  updateEmbeddingLabels(runId: string, embeddingIds: string[], newLabel: string): Observable<{ updated: number }> {
    return this.http.post<{ updated: number }>(`/api/train/runs/${encodeURIComponent(runId)}/embeddings/labels`, {
      embedding_ids: embeddingIds,
      label: newLabel
    });
  }

  commitToMyThings(runId: string, embeddingIds: string[]): Observable<{ committed: number }> {
    return this.http.post<{ committed: number }>(`/api/train/runs/${encodeURIComponent(runId)}/embeddings/commit-to-things`, {
      embedding_ids: embeddingIds
    });
  }

  /**
   * Get all embeddings from the 'things' collection
   */
  getThingsEmbeddings(limit: number = 200, offset: number = 0): Observable<{ items: EmbeddingItem[], total: number }> {
    return this.http.get<{ items: EmbeddingItem[], total: number }>(`/api/train/things?limit=${limit}&offset=${offset}`);
  }

  /**
   * Search embeddings in the 'things' collection by label
   */
  searchThingsEmbeddings(query: string, limit: number = 200, offset: number = 0): Observable<{ items: EmbeddingItem[], total: number }> {
    return this.http.get<{ items: EmbeddingItem[], total: number }>(`/api/train/things/search?query=${encodeURIComponent(query)}&limit=${limit}&offset=${offset}`);
  }

  /**
   * Get image URL for things collection (uses image_id from payload)
   */
  getThingsImageUrl(item: any): string {
    const imageId = item.payload?.image_id;
    if (imageId) {
      return `/api/train/things/image/${encodeURIComponent(imageId)}.jpg`;
    }
    // Fallback to using the Milvus ID
    return `/api/train/things/image/${encodeURIComponent(item.id)}.jpg`;
  }

  /**
   * Delete all things from the things collection and bucket
   */
  deleteAllThings(): Observable<{ message: string; deleted_embeddings: number; deleted_images: number }> {
    return this.http.delete<{ message: string; deleted_embeddings: number; deleted_images: number }>('/api/train/things');
  }

  updateThingsLabels(embeddingIds: string[], newLabel: string): Observable<{ updated: number }> {
    return this.http.post<{ updated: number }>(
      '/api/train/things/labels',
      { embedding_ids: embeddingIds, label: newLabel }
    );
  }

  deleteSelectedThings(embeddingIds: string[]): Observable<{ deleted: number }> {
    return this.http.delete<{ deleted: number }>(
      '/api/train/things/selected',
      { body: { embedding_ids: embeddingIds } }
    );
  }
}
