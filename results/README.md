# Steganography Results Analysis

This folder contains the results of testing our DWT and ILWT steganography models on 5 different image sets at 224x224 resolution. The results demonstrate the superior performance of our learnable ILWT approach compared to traditional DWT.

## Results Summary

| Metric | DWT Average | ILWT Average | ILWT Improvement |
|--------|-------------|--------------|------------------|
| Hiding PSNR (dB) | 23.61 | 31.43 | +7.82 dB |
| Recovery PSNR (dB) | 16.69 | 17.54 | +0.85 dB |
| Hiding SSIM | 0.6794 | 0.8616 | +0.1821 |
| Recovery SSIM | 0.3910 | 0.4703 | +0.0793 |

## What These Results Mean

- **PSNR (Peak Signal-to-Noise Ratio)**: Higher values indicate better quality
  - Hiding PSNR measures how similar the stego image is to the original host image (higher = better invisibility)
  - Recovery PSNR measures how well the secret image was recovered (higher = better recovery)

- **SSIM (Structural Similarity Index)**: Higher values indicate better structural similarity
  - Measures perceived quality based on luminance, contrast, and structure

- **Key Findings**:
  - ILWT consistently outperforms DWT in all metrics across all 5 test sets
  - The average hiding PSNR improvement of 7.82 dB indicates significantly better invisibility of steganography
  - The average recovery PSNR improvement of 0.85 dB indicates better secret image recovery
  - Both SSIM metrics show improvement with ILWT, confirming better quality in both directions

## Files in This Folder

- `steganography_results_set_1.png` to `steganography_results_set_5.png`: Individual visual comparisons for each image set
- `summary_results.txt`: Detailed results for each set and overall averages

## Conclusion

The learnable ILWT approach demonstrates significant improvements over traditional DWT for steganography applications, with an average hiding PSNR improvement of 7.82 dB and consistent performance across multiple image pairs. This confirms our hypothesis that learnable wavelet transforms can better adapt to image content for steganography purposes.