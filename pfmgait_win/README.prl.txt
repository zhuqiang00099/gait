On how to improve tracklet-based gait recognition systems
=========================================================

F.M. Castro and M.J. Marin-Jimenez

Contents
~~~~~~~~

This package contains Matlab code for reproducing experiments presented in [1]:
 * demoSparseTracklets: demo on how to compute sparse tracklets and apply RootDCS
 * demometric: demo on how to apply metric learning to PFM descriptors

Prerequisites
~~~~~~~~~~~~~
 * VLFeat library >=0.9.19 for Matlab: http://www.vlfeat.org/
 * Dense tracks software from [2]: http://www.irisa.fr/texmex/people/jain/w-Flow/ 

Quick start
~~~~~~~~~~~

1) Start Matlab
2) cd <pfmrootdir>
3) addpath(genpath(pwd))
4) demoSparseTracklets
5) demometric

Support
~~~~~~~

For any query/suggestion/complaint or simply to say you like/use the software, just drop us an email

fcastro<AT>uma<DOT>es
mjmarin<AT>uco<DOT>es

References
~~~~~~~~~~

[1] M. Marin-Jimenez, F.M. Castro, A. Carmona-Poyato, N. Guil, 
"On how to improve tracklet-based gait recognition systems", Pattern Recognition Letters, vol. 68, 2015, pp. 103-110

[2] M. Jain, H. Jegou, and P. Bouthemy, 
"Better exploiting motion for better action recognition", in Proc. CVPR, 2013, pp. 2555–2562

Version history
~~~~~~~~~~~~~~~
- v0: initial release
