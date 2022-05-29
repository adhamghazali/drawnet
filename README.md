![drawnet](.\images\drawnet.png)

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



TODO

- [x] T5-Encoder
- [x] Laion 400m downloader
- [x] Base Model (WIP)
- [ ] Data loader (WIP)
- [ ] Diffusion(DDIM) model
- [ ] Model Build
- [ ] Train Loop
- [ ] Structure
