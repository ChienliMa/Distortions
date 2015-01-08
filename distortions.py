#-*- coding:utf-8 -*-
"""
Implementation of some image ditortion teniques

Author: Chienli Ma
Date: 2015.01.08

Incentive:
    These distortion techniques were first designed for MNIST and grayscale.
    I modified them so that it can deal with RGB images.
    However, the performance is not very good the when dealing with RGB image.

Recommended usage:
    import distrotions
    result1 = distortions.distort( top, base, 'method' )
    result2 = distortions.distort( top, base, ['mothod_1', ... ,'method_n'])

Not recommended usage:
    import distortions
    result1 = distortions.mode_name( top, base )

Refernence:

"""
import numpy as np
import scipy.ndimage
import math
import cv2
import pdb

def distort( src, method, complex = None ):
    """
    Factory interface for distort
    """
    if complex:
        return globals().get(method)( src, complex )
    else:
        return globals().get(method)( src )

def methods():
    print """
All availale methods:
    local_elastic                  slant
    pinch                          thickness                            
    contrast                       salt_and_pepper_noise
    gaussian_noise                 uniform_noise 
    scratch                        permute
    smooth                         blur
    Lpl                            sobel
    """

def affine( src, complex = 0.7 ):
    """
    """
    w,h,c = src.shape
    theta = np.random.uniform( -90*complex,90*complex )
    scale = np.random.uniform( 1-0.5*complex, 1+0.5*complex)
    center = ( src.shape[0]//2 , src.shape[1]//2 )
    
    mat = cv2.getRotationMatrix2D( center, theta, scale )
    mat += np.random.uniform( -0.1*complex, 0.1*complex, 6).reshape([2,3])
    return cv2.warpAffine( src, mat, (w,h) )

def local_elastic( src, complex = 0.7 ):
    """
    todo: reimplementation in cyyhon
    """        
    w,h,c = src.shape
    # generate initial change displacement fields
    field = np.random.uniform( -1, 1, w*h*2 )
    field = field.reshape( [ w, h, 2 ] ) 

    # gaussian convolution, we use kernel size of 5
    field = cv2.GaussianBlur( field, (9,9), 5-3*math.pow(complex, 1./3.0))
    
    # normalize
    field  -= field.min()
    field  /= field.max() 
    # rescale
    field *= 8*math.pow(complex, 1/3.0)
    # displace
    dst = src.copy()    
    for x in xrange( w ):
        for y in xrange( h ):
            x_dst = x + field[x,y,0]
            y_dst = y + field[x,y,1]
            # deal with corner case
            if x_dst<0 or x_dst>dst.shape[0]-2 or y_dst<0 or y_dst>dst.shape[1]-2:
                dst[x,y,:] = 0
                continue
            # exchange value
            x1 = np.floor(x_dst)
            y1 = np.floor(y_dst)
                     
            dst[x,y] = src[x1:x1+1, y1:y1+1,:].mean(axis=0).mean(axis=0)
    return  dst

def slant( src, complex = 0.7 ):
    slant = np.random.uniform( 0, complex )
    w, h ,c = src.shape
    dst = np.zeros( [ w , 3*h, c ] )
    for i in xrange(w):
        start = h + np.floor( slant * i )
        dst[ i, start:start+h ] = src[i,:]
    center = h + w * slant / 2
    return dst[ :, center : center+h ]

# elements for thickness method
elements = []
elements.append( cv2.getStructuringElement( cv2.MORPH_RECT,(1,3)) )
elements.append( cv2.getStructuringElement( cv2.MORPH_RECT,(3,1)) )
elements.append( cv2.getStructuringElement( cv2.MORPH_RECT,(3,3)) )
elements.append( cv2.getStructuringElement( cv2.MORPH_CROSS,(3,3)) )
elements.append( cv2.getStructuringElement( cv2.MORPH_RECT,(3,5)) )
elements.append( cv2.getStructuringElement( cv2.MORPH_RECT,(5,3)) )
elements.append( cv2.getStructuringElement( cv2.MORPH_CROSS,(3,5) ))
elements.append( cv2.getStructuringElement( cv2.MORPH_CROSS,(5,3)))
elements.append( cv2.getStructuringElement( cv2.MORPH_RECT, (5,5)))
elements.append( cv2.getStructuringElement( cv2.MORPH_CROSS, (5,5)))

def thickness( img, complex = 0.7 ):
    """
    """
    # need to maintain a cache
    is_dilate = np.random.binomial(1,0.5)
    # dilate with random element
    index = np.random.randint(0,4)
    element = elements[ index ]
    dst= cv2.dilate( img, element ) 
    # erode with random element
    index = np.random.randint(0,8)
    element = elements[ index ]
    return cv2.erode( dst, element )
    
def pinch( src, complex = 0.7 ):
    """
    并不是理想的实现方法，理想的实现方法只用遍历半幅甚至1/4幅图像
    此外没有使用线性差值方法
    BUG：在dist ～= 半径的时候会出现问题
    性能瓶颈：两个for循环，可以考虑用C写。
    """    
    w,h,c = src.shape
    radius = int(max(w,h)*0.3)
    # center x, center y
    cx = np.floor( w/2 )
    cy = np.floor( h/2 )
    
    # get random pinch and some other thing
    pinch = np.random.uniform( -1*complex, 0.7*complex )
    radius2 = radius ** 2
    
    # destination image
    dst = np.copy( src )
    for x in xrange( w ):
        for y in xrange( h ):
            dist = (x-cx)**2 + (y-cy)**2
            # if inside disk, do some calculation
            if dist == 0:
                dst[ x, y ] = src[ cx, cy ]
            elif dist <= radius2:
                d1 = np.sqrt( dist )/float(radius)
                base = math.sin( math.pi * d1 / 2)
                d2 = math.pow( base, -pinch) 
                
                #线性插值实现
                dx =  (x -cx)*d2                     
                dy =  (y -cy)*d2      
                
                x1 = np.floor(dx)
                x2 = np.ceil(dx)
                y1 = np.floor(dy)
                y2 = np.ceil(dy)
                
                v1 = src[cx+x1,cy+y1]
                v2 = src[cx+x2,cy+y1]
                v3 = src[cx+x1,cy+y2]
                v4 = src[cx+x2,cy+y2]
                
                dx = dx%1.0
                dy = dy%1.0
                
                value = dy*(dx*v1+(1-dx)*v2) + (1-dy)*(dx*v3+(1-dx)*v4)  
                dst[x, y] = value
    return dst
    
def smooth( src, complex = 0.7 ):
    """
    Firstly, smooth the image using gaussian filter
    Second, use different mask to sample filtered image
    Third, return the average of sampled images and origin image
    """
    w,h,c = src.shape
    # deep copy
    dst = src.copy().astype('float')
    
    # 缩小卷积核的大小为原来的0.5
    k_size = int( np.random.uniform( w//5*2 ,h//5*2 + 20*complex) )
    if k_size % 2 == 0:
        k_size += 1
    # smooth
    sigma = np.random.uniform(2, 2+6*complex)
    return cv2.GaussianBlur( dst, (k_size, k_size), np.sqrt(sigma) )

    
def permute( src, complex = 0.7 ):
    """
    """    
    dst = src.copy()
        
    num_of_pix = dst.shape[0]*dst.shape[1]
    num_of_selected = int( num_of_pix * 0.2 * complex )
    for x in np.random.randint( 1, dst.shape[0] - 1, num_of_selected ):
        y = np.random.randint( 1, dst.shape[1] - 1 )
        case = np.random.randint(0,4)
        if case == 0:           # case left
            dst[x,y] = src[x-1,y]
        elif case == 1:         # case right
            dst[x,y] = src[x+1,y]
        elif case == 2:         # case top
            dst[x,y] = src[x,y+1]
        elif case == 3:         # case buttom
            dst[x,y] = src[x,y-1]
    return dst
            
def gaussian_noise( src, complex = 0.7 ):
    """
    """
    w,h,c = src.shape 
    is_float = src.dtype == np.float
    dst = src.copy()
    # noise
    noise = np.random.normal( 0, complex/10.0 , src.size ).reshape(src.shape)
    # normalize
    dst = dst / float( dst.max() ) + noise
    dst -= dst.min()
    dst /= dst.max()
    # if is 8-bit map
    if not is_float:
        dst = (dst*255).astype('uint8')
    return  dst
    
def uniform_noise( src, complex = 0.7 ):    
    """
    Can not apply to color image.
    Something need to be define
    """
    w,h,c = src.shape
    dst = src.copy()            
    num_of_pix = dst.size
    num_of_selected = int( num_of_pix * 0.2 * complex )
    is_float = src.dtype == np.float
    if is_float:
        for x in np.random.randint(0, w, num_of_selected ):
            y = np.random.randint(0, h)
            dst[x,y] = np.random.unifom(0,complex,c)
    else:
        for x in np.random.randint(0, w, num_of_selected ):
            y = np.random.randint(0, h)
            dst[x,y,:] = np.random.randint(0,complex*255+1,c)
    return dst

def salt_and_pepper_noise( src, complex = 0.7 ):
    """
    """
    w,h,c = src.shape
    dst = src.copy()            
    num_of_pix = dst.size
    num_of_selected = int( num_of_pix * 0.2 * complex )
    is_float = src.dtype == np.float
    if is_float:
        for x in np.random.randint(0, w, num_of_selected ):
            y = np.random.randint(0, h)
            dst[x,y] = np.random.randint(0,2,c)
    else:
        for x in np.random.randint(0, w, num_of_selected ):
            y = np.random.randint(0, h)
            dst[x,y,:] = np.random.randint(0,2,c)*255
    return dst

def contrast( src, complex = 0.7 ):
    """
    when do we need copy?
    """
    dst = src.copy()
    c = np.random.uniform( 1- 0.85*complex, 1)
    dst *= c
    if dst.dtype == np.float:
        dst += (1-c)/2.0
        return dst
    else:
        dst += (1-c)/2*255
        return dst.astype('uint8')

def scratch( src, complex = 0.7 ):
    """
    """    
    w,h,c = src.shape
    scratch = np.zeros( src.shape )
    width = np.random.randint( 2,2+ 3*complex ) 
    scratch[:, h/2-width:h/2+width, :] = np.random.uniform( 0, complex, 3)
    scratch = affine( scratch, complex )
    if src.dtype == np.float:
        return np.maximum(scratch, src )
    else:
        return np.maximum( (scratch*255).astype('uint8'), src )

def blur( src, complex = 0.7 ):
    theta = np.random.uniform( 0, 360 )
    length = np.random.uniform( 1, 1+( 2 * complex )** 2)*(src.shape[1]/28.0)
    filter = get_linear_motion_blur_filter( length, theta )
    return cv2.filter2D( src, -1, filter) 
    
def get_linear_motion_blur_filter( length, theta ):
    """
    helper function.
    given length and theta,return a linear motion blur filter
    """
    if length == 0:
        return np.array([1])
    
    # special angle
    if theta == 0:
        length = np.ceil( length )
        kernel = np.zeros( [length*2-1, length*2-1] )
        kernel[length, length: ] = 1./length
    elif theta == 90:            
        length = np.ceil( length )
        kernel = np.zeros( [length*2-1, length*2-1] )
        kernel[ :length, length ] = 1./length
    elif theta == 180:
        length = np.ceil( length )
        kernel = np.zeros( [length*2-1, length*2-1] )
        kernel[ length, :length ] = 1./length
    elif theta == 270:
        length = np.ceil( length )
        kernel = np.zeros( [length*2-1, length*2-1] )
        kernel[ length:, length ] = 1./length
    else:
        # common angle        
        x_length = np.cos( theta ) * length 
        y_length = np.sin( theta ) * length
        half_kernel_size =  np.ceil( max( abs(x_length), abs(y_length) ) ) 
        half_kernel_size = int(half_kernel_size)
        kernel_size = half_kernel_size * 2 + 1
        kernel = np.zeros( [kernel_size, kernel_size] )
        
        if abs(x_length) >= abs(y_length):
            direc = x_length/abs(x_length)
            for i in xrange( half_kernel_size ):
                x_i = half_kernel_size + i* direc
                y_i = half_kernel_size + np.tan(theta) * i * direc 
                y_top = np.ceil( y_i )
                y_buttom = np.floor( y_i )
                kernel[x_i, y_top] = 1./length * ( y_top - y_i )
                kernel[x_i, y_buttom] = 1./length * ( y_i - y_buttom )                    
        else:
            direc = y_length/abs(y_length)
            for i in xrange( half_kernel_size ):
                y_i = half_kernel_size + i* direc
                x_i = half_kernel_size + i * direc / np.tan(theta)  
                x_right = np.ceil( x_i )
                x_left = np.floor( x_i )
                kernel[ x_right , y_i ] = 1./length * ( x_right - x_i )
                kernel[ x_left, y_i ] = 1./length * ( x_i - x_left ) 
        kernel[half_kernel_size,half_kernel_size] = 1./length
    return abs(kernel)

# not recommended for color images
def sobel( src ):
    dx = np.random.randint(0,2)
    dy = np.random.randint(0,2)
    if dx==0 and dy==0:
        dx = 1
    dst = cv2.Sobel( src, -1, dx, dy);
    if src.dtype == np.float:
        return dst / dst.max()  
    else:
        return (dst).astype('uint8') 
        
def Lpl( src ):
    dst = cv2.Laplacian( src, -1, ksize = 3)
    dst -= dst.min()
    if src.dtype == np.float:
        return dst / dst.max()  
    else:
        return (dst).astype('uint8') 
