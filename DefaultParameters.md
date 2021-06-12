## 1.9 nm D-Spacing

```python
blur_iteration          = 200                   # Best = 4 prev Best = 200
dspace_frac_Blur_kernel = int(0.09*dspace_pix)                  # Best = 0.5, prev Best 0.09, Fraction of dspace for the blur kernel size 
closing_k_size          = 15                    # Kernel Size
opening_k_size          = 17                    # Kernel Size
pixThresh               = int(0.625*dspace_pix)  # 1.25*dspace_pix,Threshold number of pixels consituting polymers
ellipse_len             = int(1.5*dspace_pix)   # Old = 160,Breaking polymer into this size before constructing ellipse 
ellipseAspectRatio      = 5                     # Threshold aspect Ratio of the ellipse
thresh_dist             = int(1.35*dspace_pix)  # Old = 250, Distance threshold for adjacency matrix 
thresh_theta            = 10                    # delta Theta threshold for adjacency matrix 
clusterSize             = 10                    # old = 10, Threshold Crystal cluster size 
```

```python
blur_iteration          = 30                    # Best = 4 prev Best = 200
dspace_frac_Blur_kernel = int(0.09*dspace_pix)   # Best = 0.5, prev Best 0.09, Fraction of dspace for the blur kernel size 
closing_k_size          = 15                    # Kernel Size
opening_k_size          = 17                    # Kernel Size
pixThresh               = int(0.625*dspace_pix) # 1.25*dspace_pix,Threshold number of pixels consituting polymers
ellipse_len             = int(1.5*dspace_pix)   # Old = 160,Breaking polymer into this size before constructing ellipse 
ellipseAspectRatio      = 5                     # Threshold aspect Ratio of the ellipse
thresh_dist             = int(2.0*dspace_pix)  # Old = int(1.35*dspace_pix), Distance threshold for adjacency matrix 
thresh_theta            = 10                    # delta Theta threshold for adjacency matrix 
clusterSize             = 10                     # old = 10, Threshold Crystal cluster size 

```

### Best till now

```python
blur_iteration          = 15                    # Best = 4 prev Best = 200
dspace_frac_Blur_kernel = int(0.15*dspace_pix)   # Best = 0.5, prev Best 0.09, Fraction of dspace for the blur kernel size 
closing_k_size          = 15                    # Kernel Size
opening_k_size          = 17                    # Kernel Size
pixThresh               = int(0.625*dspace_pix) # 1.25*dspace_pix,Threshold number of pixels consituting polymers
ellipse_len             = int(1.5*dspace_pix)   # Old = 160,Breaking polymer into this size before constructing ellipse 
ellipseAspectRatio      = 5                     # Threshold aspect Ratio of the ellipse
thresh_dist             = int(2*dspace_pix)  # Old = int(1.35*dspace_pix), Distance threshold for adjacency matrix 
thresh_theta            = 10                    # delta Theta threshold for adjacency matrix 
clusterSize             = 7
```

