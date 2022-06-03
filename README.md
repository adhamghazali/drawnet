<img src="https://github.com/adhamghazali/drawnet/blob/main/images/drawnet.png" align="right" />


## drawnet Imagen - Pytorch (under construction)

Under construction Implementation of <a href="https://gweb-research-imagen.appspot.com/">Imagen</a>, Google's Text-to-Image Neural Network that beats DALL-E2, in Pytorch. It is the new SOTA for text-to-image synthesis.



```mermaid
flowchart TD
    id1(Text)-->id2(Text Encoder)
    id2(Text Encoder) --> id3(Text to Image Diffusion Model )
    id2(Text Encoder) --> id4(Super-Resolution Model )
    id3(Text to Image Diffusion Model ) -- 64x64 --> id4(Super-Resolution Model)   
    id2(Text Encoder) --> id5(Super-Resolution Model )
    id4(Super-Resolution Model) -- 256x256 --> id5(Super-Resolution Model) --> id6(1024x1024)
 
    style id2 fill:#f9f,stroke:#333,stroke-width:4px
    style id3 fill:#f9f,stroke:#333,stroke-width:4px
    style id4 fill:#f9f,stroke:#333,stroke-width:4px
    style id5 fill:#f9f,stroke:#333,stroke-width:4px
    style id1 fill:#afb,stroke:#fff,stroke-width:2px,color:#080808 ,stroke-dasharray: 5 5
    style id6 fill:#fff,stroke:#fff,stroke-width:1px,color:#111,stroke-dasharray: 1 1
```





#### Denoising Diffusion Models

```mermaid
flowchart TD
    id1(Noisy Image)-->id2(Model) --> id3(noise)
    id4(noise) --> id6(reverse process)
    id5(noise Image) --> id6(reverse process) --> id7(denoised Image)
    
   
 
    style id2 fill:#f9f,stroke:#333,stroke-width:4px
    style id3 fill:#afb,stroke:#fff,stroke-width:2px,color:#080808 ,stroke-dasharray: 5 5
    style id4 fill:#afb,stroke:#fff,stroke-width:2px,color:#080808 ,stroke-dasharray: 5 5
    style id5 fill:#afb,stroke:#fff,stroke-width:2px,color:#080808 ,stroke-dasharray: 5 5
    style id1 fill:#afb,stroke:#fff,stroke-width:2px,color:#080808 ,stroke-dasharray: 5 5
    style id6 fill:#f9f,stroke:#333,stroke-width:4px
    style id7 fill:#afb,stroke:#fff,stroke-width:2px,color:#080808 ,stroke-dasharray: 5 5

```



##### process: 

```mermaid
flowchart LR
id1(Random Noise Xt-1)-->id2(Model+ Reverse) --> id3(denoised Image Xt)
id3(denoised Image Xt) --> id4(Model + Reverse) --........ --> id7(Image)

 style id2 fill:#f9f,stroke:#333,stroke-width:4px
 style id1 fill:#afb,stroke:#fff,stroke-width:2px,color:#080808 ,stroke-dasharray: 5 5
 style id3 fill:#afb,stroke:#fff,stroke-width:2px,color:#080808 ,stroke-dasharray: 5 5
 style id4 fill:#f9f,stroke:#333,stroke-width:4px
 style id7 fill:#afb,stroke:#fff,stroke-width:2px,color:#080808 ,stroke-dasharray: 5 5
```



#### Sample Usage (API) - WIP

```python
import drawnet

```

TODO

- [x] T5-Encoder
- [x] Laion 400m downloader
- [x] Base Model (WIP)
- [x] Data loader
- [ ] Diffusion(DDIM) model
- [ ] Model Build
- [ ] Train Loop
- [ ] Structure
