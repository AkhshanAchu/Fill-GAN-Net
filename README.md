# FILL-GAN-NET: Unsupervised Image Inpainting via Self-Attention GANs

FILL-GAN-NET is a deep learning model for **image inpainting without the use of explicit masks during training**. It leverages a **Self-Attention GAN (SAGAN)** trained on the CelebA dataset to generate realistic face images, and then applies a lightweight **adaptor network** to learn the inpainting task from incomplete inputs.

---

## Key Features

-  **Unmasked Inpainting**: No masks are required during GAN training.
-  **Adaptor Network**: Learns to fill missing regions post GAN training.
-  **SAGAN Backbone**: Enables globally coherent image generation with long-range dependencies.
-  **CelebA Dataset**: Model is trained on thousands of face images from CelebA for realistic and diverse reconstructions.
-  **Smooth Final Output**: Combines generated outputs and incomplete inputs using **Gaussian filtering** for seamless inpainting.

---

##  Architecture

```mermaid
flowchart LR
    A[Incomplete Images] --> B[Adapter] --> C[Latent Space] --> D[SAGAN] --> E[Generated Image]
    A --> F[Overlap]
    E --> F --> G[Gaussian Filter] --> H[Final Image]
    
    style A fill:#e1f5fe
    style D fill:#e8f5e8
    style H fill:#e3f2fd
```


---

##  Dependencies

This project requires only minimal dependencies:

```text
torch
numpy
```
##  Sample Results



| Masked Image | Inpainted Iamge ||
----------------------|-------------------------|
| <img width="482" alt="Screenshot 2023-07-27 094304" src="https://github.com/user-attachments/assets/41251879-139d-4bc1-9e3a-094a0c70b803" /> | <img width="482" alt="Screenshot 2023-07-27 094304" src="https://github.com/user-attachments/assets/2526d51a-6660-4c62-9c2d-ab984a7d41f8" /> |
