## GPUJPEG

A fork of CESNET GPUJPEG to provide faster JPEG decoder library using NVIDIA GPUs, particularly in multi-GPU configurations.

Here are the major features:

* multi-GPU support for decoding JPEG

## How to build libgpujpeg library
<pre><code>
mkdir build
cd build
cmake ..
make -j
</code></pre>

## How to use

First Call ReadImageGpuInit, then call ReadImageToCVGpuMat to execute image operation

Example:
<pre><code>
#include "libgpujpeg/gpujpeg.h"
int main(){
    // only once
    ReadImageGpuInit(0, 5, 32, true);
    vector<pair<string, vector<float> > > v;
    cv::gpu::GpuMat cv_gpu_image = ReadImageToCVGpuMat("input.jpg", 31, v);
}
</code></pre>

## Useful notes

The jpegtran tool is used to set restart interval to speed up JPEG image decoding.

**How to install jpegtran**
<pre><code>
wget http://www.ijg.org/files/jpegsrc.v9c.tar.gz
tar -xzvf jpegsrc.v9c.tar.gz
cd jpeg-9c/
./configure
sudo make install
</code></pre>

**How to set restart interval**
<pre><code>
jpegtran -restart 5B -outfile output.jpg input.jpg
</code></pre>
