import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';

@Component({
  selector: 'app-splash',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div *ngIf="showSplash" 
         class="fixed inset-0 bg-gradient-to-br from-blue-600 via-purple-600 to-indigo-800 flex items-center justify-center z-50 transition-opacity duration-500"
         [class.opacity-0]="isFadingOut"
         [class.opacity-100]="!isFadingOut">
      <!-- Logo in top left -->
      <div class="absolute top-6 left-6">
        <img src="assets/images/pixit-logo-white.svg" alt="Pixit" class="h-8 w-auto" />
      </div>
      
      <div class="text-center text-white">
        <!-- Main Content -->
        <div class="mb-8">
          <h1 class="text-3xl sm:text-4xl font-bold mb-2">IntelliSight</h1>
          <p class="text-lg sm:text-xl text-blue-100">Object Intelligence Platform</p>
        </div>
        
        <!-- Bounding Box Animation -->
        <div class="mb-8">
          <div class="relative w-24 h-20 sm:w-32 sm:h-24 mx-auto">
            <!-- Bounding boxes appearing and disappearing -->
            <div class="absolute inset-0 border-2 border-white/30 rounded-lg animate-pulse"></div>
            
            <!-- Person detection box -->
            <div class="absolute top-1 left-1 sm:top-2 sm:left-2 w-6 h-4 sm:w-8 sm:h-6 border border-white/60 rounded animate-ping flex items-center justify-center" style="animation-delay: 0ms">
              <svg class="w-3 h-3 sm:w-4 sm:h-4 text-white/80" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
              </svg>
            </div>
            
            <!-- Car detection box -->
            <div class="absolute top-2 right-2 sm:top-4 sm:right-3 w-5 h-3 sm:w-6 sm:h-4 border border-white/60 rounded animate-ping flex items-center justify-center" style="animation-delay: 500ms">
              <svg class="w-2 h-2 sm:w-3 sm:h-3 text-white/80" fill="currentColor" viewBox="0 0 24 24">
                <path d="M18.92 6.01C18.72 5.42 18.16 5 17.5 5h-11c-.66 0-1.21.42-1.42 1.01L3 12v8c0 .55.45 1 1 1h1c.55 0 1-.45 1-1v-1h12v1c0 .55.45 1 1 1h1c.55 0 1-.45 1-1v-8l-2.08-5.99zM6.5 16c-.83 0-1.5-.67-1.5-1.5S5.67 13 6.5 13s1.5.67 1.5 1.5S7.33 16 6.5 16zm11 0c-.83 0-1.5-.67-1.5-1.5s.67-1.5 1.5-1.5 1.5.67 1.5 1.5-.67 1.5-1.5 1.5zM5 11l1.5-4.5h11L19 11H5z"/>
              </svg>
            </div>
            
            <!-- Generic object detection box -->
            <div class="absolute bottom-2 left-3 sm:bottom-3 sm:left-4 w-8 h-4 sm:w-10 sm:h-5 border border-white/60 rounded animate-ping flex items-center justify-center" style="animation-delay: 1000ms">
              <div class="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-white/60 rounded-full"></div>
            </div>
            
            <!-- Another generic detection box -->
            <div class="absolute bottom-1 right-1 sm:bottom-2 sm:right-2 w-6 h-4 sm:w-7 sm:h-6 border border-white/60 rounded animate-ping flex items-center justify-center" style="animation-delay: 1500ms">
              <div class="w-1 h-1 sm:w-1.5 sm:h-1.5 bg-white/60 rounded"></div>
            </div>
            
            <!-- Center detection box -->
            <div class="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-10 h-6 sm:w-12 sm:h-8 border border-white/60 rounded animate-ping flex items-center justify-center" style="animation-delay: 2000ms">
              <div class="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-white/60 rounded-full"></div>
            </div>
          </div>
        </div>
        
        <!-- Continue Button -->
        <!--<button 
          (click)="continueToApp()" 
          class="px-8 py-3 bg-white/20 hover:bg-white/30 text-white font-semibold rounded-lg transition-all duration-300 backdrop-blur-sm border border-white/30">
          Get Started
        </button>-->
      </div>
    </div>
  `,
  styles: [`
    .animate-bounce {
      animation: bounce 1s infinite;
    }
    
    @keyframes bounce {
      0%, 20%, 53%, 80%, 100% {
        transform: translate3d(0,0,0);
      }
      40%, 43% {
        transform: translate3d(0, -8px, 0);
      }
      70% {
        transform: translate3d(0, -4px, 0);
      }
      90% {
        transform: translate3d(0, -2px, 0);
      }
    }
  `]
})
export class SplashComponent implements OnInit {
  showSplash = false;
  isFadingOut = false;

  constructor(private router: Router) {}

  ngOnInit(): void {
    // Check if this is the first visit
    const hasVisited = localStorage.getItem('intellisight-visited');
// const hasVisited = false
    
    if (!hasVisited) {
      this.showSplash = true;
      // Mark as visited
      localStorage.setItem('intellisight-visited', 'true');
      
      // Auto-hide after 5 seconds if user doesn't click
      setTimeout(() => {
        if (this.showSplash) {
          this.continueToApp();
        }
      }, 3000);
    }
  }

  continueToApp(): void {
    this.isFadingOut = true;
    // Wait for fade animation to complete before hiding
    setTimeout(() => {
      this.showSplash = false;
    }, 500);
  }
}
