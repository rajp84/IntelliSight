import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { SystemService, SystemConfig, TrainingParams } from '../../core/services/system.service';
import { ToastrService } from 'ngx-toastr';

// SystemConfig moved to SystemService for app-wide access

@Component({
  selector: 'app-configuration',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './configuration.component.html',
  styleUrl: './configuration.component.scss'
})
export class ConfigurationComponent implements OnInit {
  isLoading = false;
  isSaving = false;
  errorMessage = '';
  successMessage = '';
  toastVisible = false;
  private toastTimeoutId: any = null;

  config: SystemConfig = { 
    library_path: '/', 
    library_file_extensions: 'jpg,jpeg,png,gif,webp,svg,bmp,tiff,mp4,avi,mov,wmv,flv,webm,mkv,mp3,wav,flac,aac,ogg,m4a',
    hf_token: null,
    florence_model: 'microsoft/Florence-2-large',
    dinov3_model: 'facebook/dinov3-vitb16-pretrain-lvd1689m',
    dinov3_dimension: 768,
    training_params: { batch_size: 1, detect_every: 10, frame_stride: 1, max_new_tokens: 256, mosaic_cols: 3, mosaic_tile_scale: 1, resize_width: 1024, detection_debug: true, tf32: true, score_threshold: 0.15, interpolate_boxes: true, dtype_mode: 'float32', embedding_batch_size: 32 } 
  };

  constructor(private readonly systemService: SystemService, private readonly toastr: ToastrService) {}

  private getDinov3Dimension(modelId: string): number {
    // ViT backbones
    if (modelId.includes('vits16')) {
      return 384;
    } else if (modelId.includes('vitb16')) {
      return 768;
    } else if (modelId.includes('vitl16')) {
      return 1024;
    } else if (modelId.includes('vith16plus')) {
      return 1280;
    } else if (modelId.includes('vit7b16')) {
      return 4096;
    }
    // ConvNeXt backbones
    else if (modelId.includes('convnext-tiny')) {
      return 384;
    } else if (modelId.includes('convnext-small')) {
      return 512;
    } else if (modelId.includes('convnext-base')) {
      return 1024;
    } else if (modelId.includes('convnext-large')) {
      return 1536;
    }
    // Default fallback
    return 768;
  }

  onDinov3ModelChange(): void {
    this.config.dinov3_dimension = this.getDinov3Dimension(this.config.dinov3_model);
  }

  ngOnInit(): void {
    this.load();
  }

  load(): void {
    this.isLoading = true;
    this.errorMessage = '';
    this.systemService.getConfig<SystemConfig>().subscribe({
      next: (cfg) => {
        const tp = cfg?.training_params ?? {} as TrainingParams;
        this.config = {
          library_path: cfg?.library_path ?? '/',
          library_file_extensions: cfg?.library_file_extensions ?? 'jpg,jpeg,png,gif,webp,svg,bmp,tiff,mp4,avi,mov,wmv,flv,webm,mkv,mp3,wav,flac,aac,ogg,m4a',
          hf_token: (cfg as any)?.hf_token ?? null,
          florence_model: cfg?.florence_model ?? 'microsoft/Florence-2-large',
          dinov3_model: cfg?.dinov3_model ?? 'facebook/dinov3-vitb16-pretrain-lvd1689m',
          dinov3_dimension: cfg?.dinov3_dimension ?? 768,
          training_params: {
            batch_size: tp.batch_size ?? 1,
            detect_every: tp.detect_every ?? 10,
            frame_stride: tp.frame_stride ?? 1,
            max_new_tokens: tp.max_new_tokens ?? 256,
            mosaic_cols: tp.mosaic_cols ?? 3,
            mosaic_tile_scale: tp.mosaic_tile_scale ?? 1,
            resize_width: tp.resize_width ?? 1024,
            detection_debug: tp.detection_debug ?? true,
            tf32: tp.tf32 ?? true,
            score_threshold: tp.score_threshold ?? 0.15,
            interpolate_boxes: tp.interpolate_boxes ?? true,
            dtype_mode: tp.dtype_mode ?? 'float16',
            embedding_batch_size: tp.embedding_batch_size ?? 32
          }
        };
        this.isLoading = false;
      },
      error: (err) => {
        this.errorMessage = 'Failed to load configuration';
        this.isLoading = false;
        console.error(err);
      },
    });
  }

  save(): void {
    this.isSaving = true;
    this.successMessage = '';
    this.errorMessage = '';
    this.systemService.saveConfig<SystemConfig>(this.config).subscribe({
      next: () => {
        this.successMessage = 'Configuration saved';
        this.isSaving = false;
        this.toastr.success('Configuration saved');
      },
      error: (err) => {
        this.errorMessage = 'Failed to save configuration';
        this.isSaving = false;
        console.error(err);
      },
    });
  }

  // toast handled by ngx-toastr
}
