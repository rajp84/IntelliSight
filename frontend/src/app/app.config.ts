import { ApplicationConfig, APP_INITIALIZER, provideZoneChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import { provideToastr } from 'ngx-toastr';
import { provideAnimations } from '@angular/platform-browser/animations';
import { provideRouter } from '@angular/router';
import { firstValueFrom } from 'rxjs';
import { SystemService } from './core/services/system.service';

function preloadConfigFactory(systemService: SystemService) {
  return () => firstValueFrom(systemService.getConfig());
}

import { routes } from './app.routes';

export const appConfig: ApplicationConfig = {
  providers: [
    provideZoneChangeDetection({ eventCoalescing: true }),
    provideRouter(routes),
    provideHttpClient(),
    provideAnimations(),
    provideToastr({
      positionClass: 'toast-bottom-right',
      timeOut: 2000,
      closeButton: true,
      progressBar: true
    }),
    { provide: APP_INITIALIZER, useFactory: preloadConfigFactory, deps: [SystemService], multi: true }
  ]
};
