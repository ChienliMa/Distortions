# Distortions
>Image distortion tool. Implementation of some image ditortion teniques



##Incentive:
    These distortion techniques were first designed for MNIST and grayscale images.
    I modified them so that it can deal with RGB images.
    However, the performance is not very good the when dealing with RGB image.

##Recommended usage:
    import distrotions
    result1 = distortions.distort( top, base, 'method' )
    result2 = distortions.distort( top, base, ['mothod_1', ... ,'method_n'])

##Not recommended usage:
    import distortions
    result1 = distortions.mode_name( top, base )

##Refernence:
>[Deep Self-Taught Learning for Handwritten Character Recognition](http://arxiv.org/pdf/1009.3589v1.pdf "Deep Self-Taught Learning for Handwritten Character Recognition")

## Todo:
    1.Fine tune parameters.
    2.Two version, one for grayscale images, one for RGB images