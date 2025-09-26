import { Injectable, inject } from '@angular/core';
import { io, Socket } from 'socket.io-client';
import { BehaviorSubject, Observable } from 'rxjs';

@Injectable({ providedIn: 'root' })
export class SocketService {
  private socket: Socket | null = null;
  private connectedSubject = new BehaviorSubject<boolean>(false);
  readonly connected$ = this.connectedSubject.asObservable();

  connect(url: string = window.location.origin): void {
    if (this.socket) return;
    this.socket = io(url, { transports: ['websocket', 'polling'] });
    this.socket.on('connect', () => this.connectedSubject.next(true));
    this.socket.on('disconnect', () => this.connectedSubject.next(false));
  }

  on<T = unknown>(event: string, handler: (data: T) => void): void {
    this.socket?.on(event, handler as any);
  }

  emit<T = unknown>(event: string, data?: T): void {
    this.socket?.emit(event, data as any);
  }

  disconnect(): void {
    this.socket?.disconnect();
    this.socket = null;
    this.connectedSubject.next(false);
  }
}


