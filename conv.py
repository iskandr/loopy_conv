############################################################################

# To start
# source env/bin/activate

# To list OpenCL devices
# python ./pyopencl/examples/dump-properties.py -s

# To run without asking question above device
# PYOPENCL_CTX=int python demo.py

# Update Andrea's code base
#  ../repo sync

############################################################################

import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom

# setup
# -----
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# This was for manual case (when not using auto_test_vs_ref)
#img = cl.clrandom.rand(queue, (1024, 1024, 3), dtype=np.float32)
#f = cl.clrandom.rand(queue, (7, 7, 3, 17), dtype=np.float32)

# create multiple 2D kernel
# ------
knl = lp.make_kernel(ctx.devices[0],
        "{ [im_x, im_y, f_x, f_y, color, feat, n]: -f_w <= f_x,f_y <= f_w \
		and f_w <= im_x < im_w-f_w and f_w <= im_y < im_h-f_w \
                and 0<=feat<nfeature_maps and 0<=color<ncolors and 0<=n<=nimg}",
	""" 
	out[im_x-f_w, im_y-f_w,feat,n] = sum((f_x, f_y,color), \
		img[im_x-f_x, im_y-f_y,color,n] * f[f_w+f_x, f_w+f_y,color,feat])
        """,
        [
        lp.GlobalArg("img,f", np.float32, shape=lp.auto),
        "..."
        ],
        
        defines=dict(ncolors="3", nfeature_maps="17", im_w=32, im_h=32, nimg=32,f_w=3))

ref_knl = knl

##############################################################
# Play with things in here to optimize for speed

lp.split_reduction_outward(knl, "color") 

## Order of loop variables
knl = lp.set_loop_priority(knl, [
                                 "im_x_outer", 
                                 "im_y_outer",
                                 "n",
                                 "feat",
                                 "im_x_inner",
                                 "im_y_inner", 
                                 "f_x_outer",
                                 "f_y_outer",
                                 "f_x_inner",
                                 "f_y_inner"])

## Split loop into innner/outer. 2nd arg is size of inner loop
knl = lp.split_iname(knl, "im_x", 16)
knl = lp.split_iname(knl, "im_y", 16)
knl = lp.split_iname(knl, "f_x", 7)
knl = lp.split_iname(knl, "f_y", 7)
#knl = lp.split_iname(knl, "color", 3)


## specifying block/tread ordering of each varible: g.N, l.N, unr, [seq], ilp
knl = lp.tag_inames(knl, dict(im_x_inner="l.0", 
                              im_x_outer="g.0", 
                              im_y_inner="l.1", 
                              im_y_outer="g.1",
                              feat = "unr", 
                              f_x_inner = "unr",
                              f_y_inner = "unr",
                              ))


##############################################################

#knl = lp.split_iname(knl, "im_y", 16)#, inner_tag="l.1", outer_tag="g.1")

#evt, (out,) = knl(queue, img=img, f=f, f_w=3)


#### Auto test script. Compares untransformed kernel with transformed one
lp.auto_test_vs_ref(
    ref_knl, ctx, knl,
    parameters=dict())
### ncolors and nfeature_maps already hard set above.
