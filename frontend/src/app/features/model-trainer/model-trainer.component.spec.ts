import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ModelTrainerComponent } from './model-trainer.component';

describe('ModelTrainerComponent', () => {
  let component: ModelTrainerComponent;
  let fixture: ComponentFixture<ModelTrainerComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ModelTrainerComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ModelTrainerComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
