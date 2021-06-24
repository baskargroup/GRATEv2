## 1.9 nm D-Spacing

```python
blur_iteration          = 200                   # Number of Blur Iteration
dspace_frac_Blur_kernel = int(0.09*dspace_pix)  # Fraction of dspace for the blur kernel size 
closing_k_size          = 15                    # Closing Kernel Size
opening_k_size          = 17                    # Opening Kernel Size
pixThresh               = int(0.625*dspace_pix) # Threshold number of pixels consituting polymers
ellipse_len             = int(1.5*dspace_pix)   # Breaking polymer into this size before constructing ellipse 
ellipseAspectRatio      = 5                     # Threshold aspect Ratio of the ellipse
thresh_dist             = int(1.35*dspace_pix)  # Distance threshold for adjacency matrix 
thresh_theta            = 10                    # delta Theta threshold for adjacency matrix 
clusterSize             = 10                    # Threshold Crystal cluster size 
```

```python
blur_iteration          = 30                    # Number of Blur Iteration
dspace_frac_Blur_kernel = int(0.09*dspace_pix)  # Fraction of dspace for the blur kernel size 
closing_k_size          = 15                    # Closing Kernel Size
opening_k_size          = 17                    # Opening Kernel Size
pixThresh               = int(0.625*dspace_pix) # Threshold number of pixels consituting polymers
ellipse_len             = int(1.5*dspace_pix)   # Breaking polymer into this size before constructing ellipse 
ellipseAspectRatio      = 5                    	# Threshold aspect Ratio of the ellipse
thresh_dist             = int(2.0*dspace_pix)  	# Distance threshold for adjacency matrix 
thresh_theta            = 10                    # delta Theta threshold for adjacency matrix 
clusterSize             = 10                    # Threshold Crystal cluster size 
```

### Best till now

```python
blur_iteration          = 15                    # Number of Blur Iteration
dspace_frac_Blur_kernel = int(0.15*dspace_pix)  # Fraction of dspace for the blur kernel size 
closing_k_size          = 15                    # Closing Kernel Size
opening_k_size          = 17                    # Opening Kernel Size
pixThresh               = int(0.625*dspace_pix) # Threshold number of pixels consituting polymers
ellipse_len             = int(1.5*dspace_pix)   # Breaking polymer into this size before constructing ellipse 
ellipseAspectRatio      = 5                     # Threshold aspect Ratio of the ellipse
thresh_dist             = int(2*dspace_pix)  	# Distance threshold for adjacency matrix 
thresh_theta            = 10                    # delta Theta threshold for adjacency matrix 
clusterSize             = 7						# Threshold Crystal cluster size
```

