import { Routes } from '@angular/router';
import { ThingsComponent } from './features/things/things.component';
import { LibraryComponent } from './features/library/library.component';
import { ResultsComponent } from './features/results/results.component';
import { ResultDetailComponent } from './features/results/result-detail.component';
import { AnalyzeComponent } from './features/analyze/analyze.component';
import { ConfigurationComponent } from './features/configuration/configuration.component';
import { TrainerComponent } from './features/trainer/trainer.component';
import { TesterComponent } from './features/tester/tester.component';

export const routes: Routes = [
  { path: '', redirectTo: '/things', pathMatch: 'full' },
  { path: 'things', component: ThingsComponent },
  { path: 'library', component: LibraryComponent },
  { path: 'results', component: ResultsComponent },
  { path: 'results/:id', component: ResultDetailComponent },
  { path: 'analyze', component: AnalyzeComponent },
  { path: 'configuration', component: ConfigurationComponent },
  { path: 'trainer', component: TrainerComponent },
  { path: 'tester', component: TesterComponent },
];
