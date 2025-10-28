import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ThingGroupComponent } from './thing-group.component';

describe('ThingGroupComponent', () => {
  let component: ThingGroupComponent;
  let fixture: ComponentFixture<ThingGroupComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ThingGroupComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ThingGroupComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
