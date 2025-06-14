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
