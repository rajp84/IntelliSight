import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AnnotaterComponent } from './annotater.component';

describe('AnnotaterComponent', () => {
  let component: AnnotaterComponent;
  let fixture: ComponentFixture<AnnotaterComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AnnotaterComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(AnnotaterComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
