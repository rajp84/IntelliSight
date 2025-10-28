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

export interface ThingGroup {
  id: string;
  name: string;
  thumbnail_b64?: string | null;
  count: number;
}

export interface ThingGroupDetail extends ThingGroup {
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
    return this.http.get<{ items: EmbeddingItem[], total: number }>(`/api/things?limit=${limit}&offset=${offset}`);
  }

  /**
   * Search embeddings in the 'things' collection by label
   */
  searchThingsEmbeddings(query: string, limit: number = 200, offset: number = 0): Observable<{ items: EmbeddingItem[], total: number }> {
    return this.http.get<{ items: EmbeddingItem[], total: number }>(`/api/things/search?query=${encodeURIComponent(query)}&limit=${limit}&offset=${offset}`);
  }

  /**
   * Get image URL for things collection (uses image_id from payload)
   */
  getThingsImageUrl(item: any): string {
    const imageId = item.payload?.image_id;
    if (imageId) {
      return `/api/things/image/${encodeURIComponent(imageId)}.jpg`;
    }
    // Fallback to using the Milvus ID
    return `/api/things/image/${encodeURIComponent(item.id)}.jpg`;
  }

  /**
   * Delete all things from the things collection and bucket
   */
  deleteAllThings(): Observable<{ message: string; deleted_embeddings: number; deleted_images: number }> {
    return this.http.delete<{ message: string; deleted_embeddings: number; deleted_images: number }>('/api/things');
  }

  updateThingsLabels(embeddingIds: string[], newLabel: string): Observable<{ updated: number }> {
    return this.http.post<{ updated: number }>(
      '/api/things/labels',
      { embedding_ids: embeddingIds, label: newLabel }
    );
  }

  deleteSelectedThings(embeddingIds: string[]): Observable<{ deleted: number }> {
    return this.http.delete<{ deleted: number }>(
      '/api/things/selected',
      { body: { embedding_ids: embeddingIds } }
    );
  }

  // ------- Thing Groups APIs -------
  getThingGroups(): Observable<ThingGroup[]> {
    return this.http.get<ThingGroup[]>('/api/things/groups');
  }

  createThingGroup(embeddingIds: string[]): Observable<ThingGroup> {
    return this.http.post<ThingGroup>('/api/things/groups', {
      embedding_ids: embeddingIds,
    });
  }

  getThingGroup(groupId: string): Observable<ThingGroupDetail> {
    return this.http.get<ThingGroupDetail>(`/api/things/groups/${encodeURIComponent(groupId)}`);
  }

  addThingsToGroup(groupId: string, embeddingIds: string[]): Observable<{ added: number; group_id: string }> {
    return this.http.post<{ added: number; group_id: string }>(`/api/things/groups/add`, {
      group_id: groupId,
      embedding_ids: embeddingIds,
    });
  }

  removeThingsFromGroup(groupId: string, embeddingIds: string[]): Observable<{ removed: number }> {
    return this.http.post<{ removed: number }>(`/api/things/groups/remove`, {
      group_id: groupId,
      embedding_ids: embeddingIds,
    });
  }

  renameThingGroup(groupId: string, newName: string): Observable<{ id: string; name: string; updated: number }> {
    return this.http.post<{ id: string; name: string; updated: number }>(`/api/things/groups/${encodeURIComponent(groupId)}/rename`, {
      name: newName,
    });
  }

  deleteThingGroup(groupId: string): Observable<{ id: string; deleted: boolean; ungrouped_items: number }> {
    return this.http.delete<{ id: string; deleted: boolean; ungrouped_items: number }>(`/api/things/groups/${encodeURIComponent(groupId)}`);
  }

  editThingCrop(embeddingId: string, bbox: { x1: number; y1: number; x2: number; y2: number }): Observable<{ updated: number; id: string; bbox: number[] }> {
    return this.http.post<{ updated: number; id: string; bbox: number[] }>(`/api/things/${encodeURIComponent(embeddingId)}/crop`, bbox);
  }

  // ------- Negatives (Things) -------
  listNegatives(): Observable<{ items: string[] }>{
    return this.http.get<{ items: string[] }>(`/api/things/negatives/list`);
  }

  getNegativeImageUrl(objectName: string): string {
    return `/api/things/negatives/image/${encodeURIComponent(objectName)}`;
    }

  deleteNegative(objectName: string): Observable<{ deleted: boolean }>{
    return this.http.delete<{ deleted: boolean }>(`/api/things/negatives/${encodeURIComponent(objectName)}`);
  }
}
