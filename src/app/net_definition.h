#ifndef NET_DEFINITION_H
#define NET_DEFINITION_H

#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>

using namespace dlib;
/**
 * Convolutions of size 5
 */
template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;

/**
 * Convolutions of size 3
 */
template <long num_filters, typename SUBNET> using con3d  = con<num_filters,3,3,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con3  = con<num_filters,3,3,1,1,SUBNET>;

/**
 * Downsampler 8x using convolution of size 5
 */
template <typename SUBNET> using downsampler8x  = relu<bn_con<con5d<32, relu<bn_con<con5d<32, relu<bn_con<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using a_downsampler8x  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;

/**
 * Downsampler 8x using convolution of size 3
 */
template <typename SUBNET> using con3downsampler8x  = relu<bn_con<con3d<32, relu<bn_con<con3d<32, relu<bn_con<con3d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using a_con3downsampler8x  = relu<affine<con3d<32, relu<affine<con3d<32, relu<affine<con3d<16,SUBNET>>>>>>>>>;

/**
 * CNN blocks using convolution of size 5
 */
template <typename SUBNET> using rcon5  = relu<bn_con<con5<55,SUBNET>>>;
template <typename SUBNET> using a_rcon5  = relu<affine<con5<55,SUBNET>>>;

/**
 * CNN blocks using convolution of size 3
 */
template <typename SUBNET> using rcon3  = relu<bn_con<con3<55,SUBNET>>>;
template <typename SUBNET> using a_rcon3  = relu<affine<con3<55,SUBNET>>>;

/**
 * Net types using convolution of size 5. Test net type has bn_con layer changed to affine layer.
 */
using net_type = loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler8x<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;
using test_net_type = loss_mmod<con<1,9,9,1,1,a_rcon5<a_rcon5<a_rcon5<a_downsampler8x<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

/**
 * Net types using convolution of size 3. Test net type has bn_con layer changed to affine layer.
 */
//using net_type = loss_mmod<con<1,9,9,1,1,rcon3<rcon3<rcon3<con3downsampler8x<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;
//using test_net_type = loss_mmod<con<1,9,9,1,1,a_rcon3<a_rcon3<a_rcon3<a_con3downsampler8x<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;


template <typename SUBNET> using downsampler4x  = relu<bn_con<con5d<32, relu<bn_con<con5d<16, SUBNET>>>>>>;
template <typename SUBNET> using downsampler16x = relu<bn_con<con5d<55, relu<bn_con<con5d<55, relu<bn_con<con5d<32,SUBNET>>>>>>>>>;

using myNet_type = loss_mmod<con<1,9,9,1,1,rcon3<rcon3<rcon3<downsampler16x<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;


#endif //NET_DEFINITION_H
