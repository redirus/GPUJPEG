/**
 * Copyright (c) 2011, CESNET z.s.p.o
 * Copyright (c) 2011, Silicon Genome, LLC.
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <libgpujpeg/gpujpeg_decoder.h>
#include <libgpujpeg/gpujpeg_decoder_internal.h>
#include "gpujpeg_preprocessor.h"
#include "gpujpeg_dct_cpu.h"
#include "gpujpeg_dct_gpu.h"
#include "gpujpeg_huffman_cpu_decoder.h"
#include "gpujpeg_huffman_gpu_decoder.h"
#include <libgpujpeg/gpujpeg_util.h>


struct gpujpeg_decoder **d_ptr;
uint8_t** i_ptr;
bool GIDL_Init(int device_id, int restart_interval, int decoder_num, bool is_color)
{
    int image_num = decoder_num;
    d_ptr = (struct gpujpeg_decoder **)malloc(decoder_num*sizeof(struct gpujpeg_decoder *));
    i_ptr = (uint8_t**)malloc(image_num*sizeof(uint8_t*));

    int flags = 0;
    if(gpujpeg_init_device(device_id, flags)){
        return false;
    }
    struct gpujpeg_parameters param;
    gpujpeg_set_default_parameters(&param);
    param.restart_interval = restart_interval;
    param.interleaved = 1;
    param.device_id = device_id;

    struct gpujpeg_image_parameters param_image;
    gpujpeg_image_set_default_parameters(&param_image);
    // To do: Config
    param_image.width = 1600;
    param_image.height = 1600;
    for(int i=0;i<image_num;i++){
        cudaMallocHost((void**)(&(i_ptr[i])), 30000000 * sizeof(uint8_t));
    }
    if(is_color){ 
        param_image.comp_count = 3;
    }
    else{
        param_image.comp_count = 1;
    }
    for(int i=0;i<decoder_num;i++){
        d_ptr[i] = gpujpeg_decoder_create(NULL, &param);
        if(d_ptr[i] == NULL){
            return false;
        }
        struct gpujpeg_coder* coder;
        gpujpeg_decoder_init(d_ptr[i], &param, &param_image);
        coder = &(d_ptr[i]->coder);
        // Create buffers if not already created
        if (coder->data_raw == NULL) {
            if (cudaSuccess != cudaMallocHost((void**)&coder->data_raw, coder->data_raw_size * sizeof(uint8_t))) {
                return false;
            }
        }
        if (coder->d_data_raw_allocated == NULL) {
            if (cudaSuccess != cudaMalloc((void**)&coder->d_data_raw_allocated, coder->data_raw_size * sizeof(uint8_t))) {
                return false;
            }
        }
    }
    return true;
}
cv::gpu::GpuMat GIDL_ReadImageToCVGpuMat(const string& filename, int decoder_index, vector<pair<string, vector<float> > > op){
    
    struct gpujpeg_decoder* decoder = d_ptr[decoder_index];
    cv::gpu::GpuMat cv_img;
    int device_id = decoder->device_id;
    int image_size = 0;
    uint8_t* image = i_ptr[decoder_index];
    
    if ( gpujpeg_image_load_from_file(filename.c_str(), &image, &image_size, device_id) != 0 ){
        return cv_img;
    }
    
    uint8_t* decode_output = rocs_gpujpeg_decode(decoder, image, image_size, device_id);
    if(decode_output == NULL ){
        return cv_img;
    }

    cv::gpu::setDevice(device_id);
    cv::gpu::GpuMat cv_gpu_img(decoder->coder.param_image.height, decoder->coder.param_image.width, CV_8UC3, (void*)decode_output);

    std::string operation;
    std::vector<float> param;
    std::vector<pair<string, vector<float> > >::iterator op_iter;
    std::vector<float>::iterator p_iter;
    for(op_iter=op.begin();op_iter!=op.end();op_iter++){
        operation = op_iter->first;
        param = op_iter->second;
        p_iter = param.begin();
        if(operation=="rotate"){
            cv_gpu_img = ImageRotate(cv_gpu_img);
        }
        else if(operation=="resize"){
            float height = *p_iter;
            p_iter++;
            float width = *p_iter;
            cv_gpu_img = ImageResize(cv_gpu_img, (int)height, (int)width);
        }
        else if(operation=="crop"){
            float x = *p_iter;
            p_iter++;
            float y = *p_iter;
            p_iter++;
            float width = *p_iter;
            p_iter++;
            float height = *p_iter;
            cv_gpu_img = ImageCrop(cv_gpu_img, (int)x, (int)y, (int)width, (int)height);
        }    
        else{
            return cv_gpu_img;
        }
    }    
    return cv_gpu_img;
}
cv::gpu::GpuMat ImageRotate(cv::gpu::GpuMat cv_gpu_img){
    //cv::gpu::setDevice(device_id);
    cv::gpu::GpuMat cv_rotate_img;
    //cv::gpu::rotate(cv_gpu_img, cv_rotate_img, cv::Size(cv_gpu_img.cols, cv_gpu_img.rows));
    return cv_rotate_img;
}
cv::gpu::GpuMat ImageResize(cv::gpu::GpuMat cv_gpu_img, int height, int width){
    //cv::gpu::setDevice(device_id);
    cv::gpu::GpuMat cv_resize_img;
    if (height > 0 && width > 0) {
        int new_width = width;
        int new_height = height;
        if (height == 1 || width == 1) {
            float length = height > width ? height : width;
            if (cv_gpu_img.rows < cv_gpu_img.cols) {
                float scale = length / cv_gpu_img.rows;
                new_width = scale * cv_gpu_img.cols;
                new_height = length;
            }
            else {
                float scale = length / cv_gpu_img.cols;
                new_width = length;
                new_height = scale * cv_gpu_img.rows;
            }
        }
        cv::gpu::resize(cv_gpu_img, cv_resize_img, cv::Size(new_width, new_height));
    }
    return cv_resize_img;
}
cv::gpu::GpuMat ImageCrop(cv::gpu::GpuMat cv_gpu_img, int x, int y, int width, int height){
    //cv::gpu::setDevice(device_id);
    cv::gpu::GpuMat cv_crop_img;
    // Setup a rectangle to define your region of interest
    cv::Rect myROI(x, y, width, height);
    cv_crop_img = cv_gpu_img(myROI);
    return cv_crop_img;
}

/** Documented at declaration */
void
gpujpeg_decoder_output_set_default(struct gpujpeg_decoder_output* output)
{
    output->type = GPUJPEG_DECODER_OUTPUT_INTERNAL_BUFFER;
    output->data = NULL;
    output->data_size = 0;
    output->texture = NULL;
}

/** Documented at declaration */
void
gpujpeg_decoder_output_set_custom(struct gpujpeg_decoder_output* output, uint8_t* custom_buffer)
{
    output->type = GPUJPEG_DECODER_OUTPUT_CUSTOM_BUFFER;
    output->data = custom_buffer;
    output->data_size = 0;
    output->texture = NULL;
}

/** Documented at declaration */
void
gpujpeg_decoder_output_set_texture(struct gpujpeg_decoder_output* output, struct gpujpeg_opengl_texture* texture)
{
    output->type = GPUJPEG_DECODER_OUTPUT_OPENGL_TEXTURE;
    output->data = NULL;
    output->data_size = 0;
    output->texture = texture;
}

/** Documented at declaration */
void
gpujpeg_decoder_output_set_cuda_buffer(struct gpujpeg_decoder_output* output)
{
    output->type = GPUJPEG_DECODER_OUTPUT_CUDA_BUFFER;
    output->data = NULL;
    output->data_size = 0;
    output->texture = NULL;
}

/** Documented at declaration */
void
gpujpeg_decoder_output_set_custom_cuda(struct gpujpeg_decoder_output* output, uint8_t* d_custom_buffer)
{
    output->type = GPUJPEG_DECODER_OUTPUT_CUSTOM_CUDA_BUFFER;
    output->data = d_custom_buffer;
    output->data_size = 0;
    output->texture = NULL;
}

/** Documented at declaration */
struct gpujpeg_decoder*
gpujpeg_decoder_create(cudaStream_t * stream, struct gpujpeg_parameters* param)
{
    struct gpujpeg_decoder* decoder = (struct gpujpeg_decoder*) malloc(sizeof(struct gpujpeg_decoder));
    if ( decoder == NULL )
        return NULL;

    // Get coder
    struct gpujpeg_coder* coder = &decoder->coder;

    // Set parameters
    memset(decoder, 0, sizeof(struct gpujpeg_decoder));
    gpujpeg_set_default_parameters(&coder->param);
    gpujpeg_image_set_default_parameters(&coder->param_image);
    coder->param_image.comp_count = 0;
    coder->param_image.width = 0;
    coder->param_image.height = 0;
    coder->param.restart_interval = 0;

    int result = 1;

    // To do: Check
    cudaSetDevice(param->device_id);
    // Create reader
    decoder->reader = gpujpeg_reader_create();
    if ( decoder->reader == NULL )
        result = 0;

    // Allocate quantization tables in device memory
    for ( int comp_type = 0; comp_type < GPUJPEG_COMPONENT_TYPE_COUNT; comp_type++ ) {
        if ( cudaSuccess != cudaMalloc((void**)&decoder->table_quantization[comp_type].d_table, 64 * sizeof(uint16_t)) )
            result = 0;
    }
    // Allocate huffman tables in device memory
    for ( int comp_type = 0; comp_type < GPUJPEG_COMPONENT_TYPE_COUNT; comp_type++ ) {
        for ( int huff_type = 0; huff_type < GPUJPEG_HUFFMAN_TYPE_COUNT; huff_type++ ) {
            if ( cudaSuccess != cudaMalloc((void**)&decoder->d_table_huffman[comp_type][huff_type], sizeof(struct gpujpeg_table_huffman_decoder)) )
                result = 0;
        }
    }
    gpujpeg_cuda_check_error("Decoder table allocation", return NULL);

    // Init huffman encoder
    if (gpujpeg_huffman_gpu_decoder_init() != 0) {
        result = 0;
    }

    // Stream
    decoder->stream = stream;
    if (decoder->stream == NULL) {
        decoder->allocatedStream = (cudaStream_t *) malloc(sizeof(cudaStream_t));
        // if (cudaSuccess != cudaStreamCreate(decoder->allocatedStream)) {
        // To do: Check
        if (cudaSuccess != cudaStreamCreateWithFlags(decoder->allocatedStream, cudaStreamNonBlocking)) {
            result = 0;
        }
        decoder->stream = decoder->allocatedStream;
    }

    if (result == 0) {
        gpujpeg_decoder_destroy(decoder);
        return NULL;
    }

    decoder->device_id = param->device_id;

    return decoder;
}

/** Documented at declaration */
int
gpujpeg_decoder_init(struct gpujpeg_decoder* decoder, struct gpujpeg_parameters* param, struct gpujpeg_image_parameters* param_image)
{
    assert(param_image->comp_count == 1 || param_image->comp_count == 3);

    // Get coder
    struct gpujpeg_coder* coder = &decoder->coder;

    // Check if (re)inialization is needed
    int change = 0;
    change |= coder->param_image.width != param_image->width;
    change |= coder->param_image.height != param_image->height;
    change |= coder->param_image.comp_count != param_image->comp_count;
    change |= coder->param.restart_interval != param->restart_interval;
    change |= coder->param.interleaved != param->interleaved;
    for ( int comp = 0; comp < param_image->comp_count; comp++ ) {
        change |= coder->param.sampling_factor[comp].horizontal != param->sampling_factor[comp].horizontal;
        change |= coder->param.sampling_factor[comp].vertical != param->sampling_factor[comp].vertical;
    }
    if ( change == 0 )
        return 0;

    // For now we can't reinitialize decoder, we can only do first initialization
    if ( coder->param_image.width != 0 || coder->param_image.height != 0 || coder->param_image.comp_count != 0 ) {
        // To do: Check
        // fprintf(stderr, "[GPUJPEG] [Error] Can't reinitialize decoder, implement if needed!\n");
        return -1;
    }

    // Initialize coder
    if ( gpujpeg_coder_init(coder) != 0 ) {
        return -1;
    }
    if (0 == gpujpeg_coder_init_image(coder, param, param_image, decoder->stream)) {
        return -1;
    }

    // Init postprocessor
    if ( gpujpeg_preprocessor_decoder_init(&decoder->coder) != 0 ) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed to init postprocessor!\n");
        return -1;
    }

    return 0;
}

uint8_t*
rocs_gpujpeg_decode(struct gpujpeg_decoder* decoder, uint8_t* image, int image_size, int device_id)
{
    cudaSetDevice(device_id);

    // Get coder
    struct gpujpeg_coder* coder = &decoder->coder;

    // Start copying compressed data to device memory
    decoder->data_compressed = image;
    decoder->data_compressed_size = image_size;
    if (decoder->data_compressed_size > coder->data_compressed_allocated_size) {
        if (coder->d_data_compressed != NULL) {
            cudaFree(coder->d_data_compressed);
            coder->d_data_compressed = NULL;
        }
        cudaMalloc((void**)&coder->d_data_compressed, decoder->data_compressed_size * sizeof(uint8_t));
        gpujpeg_cuda_check_error("Coder data compressed device allocation", return NULL);
        coder->data_compressed_allocated_size = decoder->data_compressed_size;
    }
    cudaMemcpyAsync(coder->d_data_compressed, decoder->data_compressed, decoder->data_compressed_size * sizeof(uint8_t), cudaMemcpyHostToDevice, *(decoder->stream));
    gpujpeg_cuda_check_error("Decoder copy compressed data", return NULL);

    // Read JPEG image data
    if (0 != gpujpeg_reader_read_image(decoder, image, image_size)) {
        fprintf(stderr, "[GPUJPEG] [Error] Decoder failed when decoding image data!\n");
        return NULL;
    }

    // Perform huffman decoding on CPU (when there are not enough segments to saturate GPU)
    if (coder->segment_count < 256) {
        printf("Warning: Perform huffman decoding on CPU");
        if (0 != gpujpeg_huffman_cpu_decoder_decode(decoder)) {
            fprintf(stderr, "[GPUJPEG] [Error] Huffman decoder failed!\n");
            return NULL;
        }
        cudaStreamSynchronize(*(decoder->stream));
        // Copy quantized data to device memory from cpu memory
        cudaMemcpyAsync(coder->d_data_quantized, coder->data_quantized, coder->data_size * sizeof(int16_t), cudaMemcpyHostToDevice, *(decoder->stream));
        cudaStreamSynchronize(*(decoder->stream));
        
        GPUJPEG_CUSTOM_TIMER_START(decoder->in_gpu);
    }
    // Perform huffman decoding on GPU (when there are enough segments to saturate GPU)
    else {
        // printf("Perform huffman decoding on GPU!\n");

        // Reset huffman output
        cudaMemsetAsync(coder->d_data_quantized, 0, coder->data_size * sizeof(int16_t), *(decoder->stream));
        
        // Copy segments to device memory
        cudaMemcpyAsync(coder->d_segment, coder->segment, decoder->segment_count * sizeof(struct gpujpeg_segment), cudaMemcpyHostToDevice, *(decoder->stream));
        gpujpeg_cuda_check_error("Decoder copy compressed data", return NULL);
        
        // Zero output memory
        cudaMemsetAsync(coder->d_data_quantized, 0, coder->data_size * sizeof(int16_t), *(decoder->stream));
        
        // Perform huffman decoding
        if (0 != gpujpeg_huffman_gpu_decoder_decode(decoder)) {
            fprintf(stderr, "[GPUJPEG] [Error] Huffman decoder on GPU failed!\n");
            return NULL;
        }
    }

    // Perform IDCT and dequantization (own CUDA implementation)
    if (0 != gpujpeg_idct_gpu(decoder)) {
        return NULL;
    }

    // Use internal CUDA buffer as decoding destination
    coder->d_data_raw = coder->d_data_raw_allocated;
    
    // Preprocessing
    if (0 != gpujpeg_preprocessor_decode(&decoder->coder, *(decoder->stream))) {
        return NULL;
    }
   
    // Wait for async operations before copying from the device
    cudaStreamSynchronize(*(decoder->stream));

    // Copy decompressed image to host memory
    //cudaMemcpy(coder->data_raw, coder->d_data_raw, coder->data_raw_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    //cudaFreeHost(image);
    return coder->d_data_raw;
    //return output->data;
}

/** Documented at declaration */
int
gpujpeg_decoder_decode(struct gpujpeg_decoder* decoder, uint8_t* image, int image_size, struct gpujpeg_decoder_output* output)
{
    // Get coder
    struct gpujpeg_coder* coder = &decoder->coder;

    // Reset durations
    coder->duration_memory_map = 0.0;
    coder->duration_memory_unmap = 0.0;
    coder->duration_stream = 0.0;
    coder->duration_in_gpu = 0.0;
    coder->duration_waiting = 0.0;

    // Start copying compressed data to device memory
    decoder->data_compressed = image;
    decoder->data_compressed_size = image_size;
    if (decoder->data_compressed_size > coder->data_compressed_allocated_size) {
        if (coder->d_data_compressed != NULL) {
            cudaFree(coder->d_data_compressed);
            coder->d_data_compressed = NULL;
        }
        cudaMalloc((void**)&coder->d_data_compressed, decoder->data_compressed_size * sizeof(uint8_t));
        gpujpeg_cuda_check_error("Coder data compressed device allocation", return 0);
        coder->data_compressed_allocated_size = decoder->data_compressed_size;
    }
    cudaMemcpyAsync(coder->d_data_compressed, decoder->data_compressed, decoder->data_compressed_size * sizeof(uint8_t), cudaMemcpyHostToDevice, *(decoder->stream));
    gpujpeg_cuda_check_error("Decoder copy compressed data", return -1);

    GPUJPEG_CUSTOM_TIMER_START(decoder->def);

    // Read JPEG image data
    if (0 != gpujpeg_reader_read_image(decoder, image, image_size)) {
        fprintf(stderr, "[GPUJPEG] [Error] Decoder failed when decoding image data!\n");
        return -1;
    }

    GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
    coder->duration_stream = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);

    // Perform huffman decoding on CPU (when there are not enough segments to saturate GPU)
    if (coder->segment_count < 256) {
        if (0 != gpujpeg_huffman_cpu_decoder_decode(decoder)) {
            fprintf(stderr, "[GPUJPEG] [Error] Huffman decoder failed!\n");
            return -1;
        }

        // Copy quantized data to device memory from cpu memory
        cudaMemcpyAsync(coder->d_data_quantized, coder->data_quantized, coder->data_size * sizeof(int16_t), cudaMemcpyHostToDevice, *(decoder->stream));

        GPUJPEG_CUSTOM_TIMER_START(decoder->in_gpu);
    }
    // Perform huffman decoding on GPU (when there are enough segments to saturate GPU)
    else {
        GPUJPEG_CUSTOM_TIMER_START(decoder->in_gpu);

        // Reset huffman output
        cudaMemsetAsync(coder->d_data_quantized, 0, coder->data_size * sizeof(int16_t), *(decoder->stream));

        // Copy segments to device memory
        cudaMemcpyAsync(coder->d_segment, coder->segment, decoder->segment_count * sizeof(struct gpujpeg_segment), cudaMemcpyHostToDevice, *(decoder->stream));
        gpujpeg_cuda_check_error("Decoder copy compressed data", return -1);

        // Zero output memory
        cudaMemsetAsync(coder->d_data_quantized, 0, coder->data_size * sizeof(int16_t), *(decoder->stream));

        // Perform huffman decoding
        if (0 != gpujpeg_huffman_gpu_decoder_decode(decoder)) {
            fprintf(stderr, "[GPUJPEG] [Error] Huffman decoder on GPU failed!\n");
            return -1;
        }
    }

    // Perform IDCT and dequantization (own CUDA implementation)
    if (0 != gpujpeg_idct_gpu(decoder)) {
        return -1;
    }

    // Create buffers if not already created
    if (coder->data_raw == NULL) {
        if (cudaSuccess != cudaMallocHost((void**)&coder->data_raw, coder->data_raw_size * sizeof(uint8_t))) {
            return -1;
        }
    }
    if (coder->d_data_raw_allocated == NULL) {
        if (cudaSuccess != cudaMalloc((void**)&coder->d_data_raw_allocated, coder->data_raw_size * sizeof(uint8_t))) {
            return -1;
        }
    }

    // Select CUDA output buffer
    if (output->type == GPUJPEG_DECODER_OUTPUT_CUSTOM_CUDA_BUFFER) {
        // Image should be directly decoded into custom CUDA buffer
        coder->d_data_raw = output->data;
    }
    else if (output->type == GPUJPEG_DECODER_OUTPUT_OPENGL_TEXTURE && output->texture->texture_callback_attach_opengl == NULL) {
        GPUJPEG_CUSTOM_TIMER_START(decoder->def);

        // Use OpenGL texture as decoding destination
        int data_size = 0;
        uint8_t* d_data = gpujpeg_opengl_texture_map(output->texture, &data_size);
        assert(data_size == coder->data_raw_size);
        coder->d_data_raw = d_data;

        GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
        coder->duration_memory_map = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);
    }
    else {
        // Use internal CUDA buffer as decoding destination
        coder->d_data_raw = coder->d_data_raw_allocated;
    }

    // Preprocessing
    if (0 != gpujpeg_preprocessor_decode(&decoder->coder, *(decoder->stream))) {
        return -1;
    }

    // Wait for async operations before copying from the device
    GPUJPEG_CUSTOM_TIMER_START(decoder->def);
    cudaStreamSynchronize(*(decoder->stream));
    GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
    coder->duration_waiting = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);

    GPUJPEG_CUSTOM_TIMER_STOP(decoder->in_gpu);
    coder->duration_in_gpu = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->in_gpu);

    // Set decompressed image size
    output->data_size = coder->data_raw_size * sizeof(uint8_t);

    // Set decompressed image
    if (output->type == GPUJPEG_DECODER_OUTPUT_INTERNAL_BUFFER) {
        GPUJPEG_CUSTOM_TIMER_START(decoder->def);

        // Copy decompressed image to host memory
        cudaMemcpy(coder->data_raw, coder->d_data_raw, coder->data_raw_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);

        GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
        coder->duration_memory_from = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);

        // Set output to internal buffer
        output->data = coder->data_raw;
    }
    else if (output->type == GPUJPEG_DECODER_OUTPUT_CUSTOM_BUFFER) {
        GPUJPEG_CUSTOM_TIMER_START(decoder->def);

        assert(output->data != NULL);

        // Copy decompressed image to host memory
        cudaMemcpy(output->data, coder->d_data_raw, coder->data_raw_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);

        GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
        coder->duration_memory_from = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);
    }
    else if (output->type == GPUJPEG_DECODER_OUTPUT_OPENGL_TEXTURE) {
        // If OpenGL texture wasn't mapped and used directly for decoding into it
        if (output->texture->texture_callback_attach_opengl != NULL) {
            GPUJPEG_CUSTOM_TIMER_START(decoder->def);

            // Map OpenGL texture
            int data_size = 0;
            uint8_t* d_data = gpujpeg_opengl_texture_map(output->texture, &data_size);
            assert(data_size == coder->data_raw_size);

            GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
            coder->duration_memory_map = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);

            GPUJPEG_CUSTOM_TIMER_START(decoder->def);

            // Copy decompressed image to texture pixel buffer object device data
            cudaMemcpy(d_data, coder->d_data_raw, coder->data_raw_size * sizeof(uint8_t), cudaMemcpyDeviceToDevice);

            GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
            coder->duration_memory_from = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);
        }

        GPUJPEG_CUSTOM_TIMER_START(decoder->def);

        // Unmap OpenGL texture
        gpujpeg_opengl_texture_unmap(output->texture);

        GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
        coder->duration_memory_unmap = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);
    }
    else if (output->type == GPUJPEG_DECODER_OUTPUT_CUDA_BUFFER) {
        // Copy decompressed image to texture pixel buffer object device data
        output->data = coder->d_data_raw;
    }
    else if (output->type == GPUJPEG_DECODER_OUTPUT_CUSTOM_CUDA_BUFFER) {
        // Image was already directly decoded into custom CUDA buffer
        output->data = coder->d_data_raw;
    }
    else {
        // Unknown output type
        assert(0);
    }

    return 0;
}

void
gpujpeg_decoder_set_output_format(struct gpujpeg_decoder* decoder,
                enum gpujpeg_color_space color_space,
                enum gpujpeg_pixel_format sampling_factor)
{
        decoder->coder.param_image.color_space = color_space;
        decoder->coder.param_image.pixel_format = sampling_factor;
}

/** Documented at declaration */
int
gpujpeg_decoder_destroy(struct gpujpeg_decoder* decoder)
{
    assert(decoder != NULL);

    if (0 != gpujpeg_coder_deinit(&decoder->coder)) {
        return -1;
    }

    for (int comp_type = 0; comp_type < GPUJPEG_COMPONENT_TYPE_COUNT; comp_type++) {
        if (decoder->table_quantization[comp_type].d_table != NULL) {
            cudaFree(decoder->table_quantization[comp_type].d_table);
        }
    }

    if (decoder->reader != NULL) {
        gpujpeg_reader_destroy(decoder->reader);
    }

    if (decoder->allocatedStream != NULL) {
        cudaStreamDestroy(*(decoder->allocatedStream));
        free(decoder->allocatedStream);
        decoder->allocatedStream = NULL;
        decoder->stream = NULL;
    }

    free(decoder);

    return 0;
}
