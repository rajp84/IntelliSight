import { Routes } from '@angular/router';
import { ThingsComponent } from './features/things/things.component';
import { LibraryComponent } from './features/library/library.component';
import { ThingGroupComponent } from './features/things/thing-group.component';
import { ResultsComponent } from './features/results/results.component';
import { ResultDetailComponent } from './features/results/result-detail.component';
import { AnalyzeComponent } from './features/analyze/analyze.component';
import { ConfigurationComponent } from './features/configuration/configuration.component';
import { TrainerComponent } from './features/trainer/trainer.component';
import { AnnotaterComponent } from './features/annotater/annotater.component';
import { ModelTrainerComponent } from './features/model-trainer/model-trainer.component';
import { JobsComponent } from './features/jobs/jobs.component';
import { JobDetailsComponent } from './features/jobs/job-details/job-details.component';
import { ModelManagerComponent } from './features/model-manager/model-manager.component';

export const routes: Routes = [
  { path: '', redirectTo: '/things', pathMatch: 'full' },
  { path: 'things', component: ThingsComponent },
  { path: 'things/groups/:groupId', component: ThingGroupComponent },
  { path: 'library', component: LibraryComponent },
  { path: 'results', component: ResultsComponent },
  { path: 'results/:id', component: ResultDetailComponent },
  { path: 'analyze', component: AnalyzeComponent },
  { path: 'configuration', component: ConfigurationComponent },
  { path: 'trainer', component: TrainerComponent },
  { path: 'jobs', component: JobsComponent },
  { path: 'jobs/:jobId', component: JobDetailsComponent },
  { path: 'annotater', component: AnnotaterComponent },
  { path: 'model-trainer', component: ModelTrainerComponent },
  { path: 'model-manager', component: ModelManagerComponent },
];
