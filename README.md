Python scripts on various topics (dynamics, seismic image denoising, number theory):
- Population dynamics & Feigenbaum diagram
- Prime numbers spiral
- Fermat's theorem and pi, li and x/ln(x) functions
- Seismic data analysis (Sleipner C02 injections) on 1994, 2001, 2004, 2008 proceedings. Note that this script require picture data to be executed. You can ask for this data at mailto:ferdinand.equilbey@etu.univ-poitiers.fr
  - seismic1.py contains picture data visualisation, noise and time variations plots, noise normality test
  - seismic2.py contains slice signals visualisation, hilbert transform, picture denoising (Gaussian kernel and Total Variation), influence of denoising on PSNR and Hilbert transform. For more information on TV denoising, you can see A. Chambolle https://www.ipol.im/pub/art/2013/61/article_lr.pdf or scikit-image documentation.
