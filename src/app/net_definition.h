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
 * Downsampler 8x using convolution of size 5
 */
template <typename SUBNET> using downsampler8x  = relu<bn_con<con5d<32, relu<bn_con<con5d<32, relu<bn_con<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using a_downsampler8x  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;

/**
 * CNN blocks using convolution of size 5, original filter size is 55
 */
template <typename SUBNET> using rcon5_55  = relu<bn_con<con5<55,SUBNET>>>;
template <typename SUBNET> using a_rcon5_55  = relu<affine<con5<55,SUBNET>>>;

template <typename SUBNET> using rcon5_50  = relu<bn_con<con5<50,SUBNET>>>;
template <typename SUBNET> using a_rcon5_50  = relu<affine<con5<50,SUBNET>>>;
template <typename SUBNET> using rcon5_40  = relu<bn_con<con5<40,SUBNET>>>;
template <typename SUBNET> using a_rcon5_40  = relu<affine<con5<40,SUBNET>>>;
template <typename SUBNET> using rcon5_32  = relu<bn_con<con5<32,SUBNET>>>;
template <typename SUBNET> using a_rcon5_32  = relu<affine<con5<32,SUBNET>>>;

/**
 * Net types using convolution of size 5. Test net type has bn_con layer changed to affine layer.
 */
using net_type = loss_mmod<con<1,9,9,1,1,rcon5_55<rcon5_55<rcon5_55<rcon5_40<downsampler8x<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>>;
using test_net_type = loss_mmod<con<1,9,9,1,1,a_rcon5_55<a_rcon5_55<a_rcon5_55<a_rcon5_40<a_downsampler8x<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>>;

template <long num_filters, typename SUBNET> using con3  = con<num_filters,3,3,1,1,SUBNET>;
template <typename SUBNET> using rcon3  = relu<bn_con<con3<55, SUBNET>>>;
template <typename SUBNET> using rcon32  = relu<bn_con<con3<40, SUBNET>>>;
template <typename SUBNET> using arcon3  = relu<affine<con3<55, SUBNET>>>;
template <typename SUBNET> using arcon32  = relu<affine<con3<40, SUBNET>>>;
using state_net_type = loss_mmod<con<1,9,9,1,1,rcon3<rcon32<input_rgb_image_pyramid<pyramid_down<2>>>>>>;
using state_test_net_type = loss_mmod<con<1,9,9,1,1,arcon3<arcon32<input_rgb_image_pyramid<pyramid_down<2>>>>>>;

//using state_net_type = loss_mmod<con<1,9,9,1,1,rcon5_55<rcon5_55<rcon5_40<input_rgb_image_pyramid<pyramid_down<3>>>>>>>;
//using state_test_net_type = loss_mmod<con<1,9,9,1,1,a_rcon5_55<a_rcon5_55<a_rcon5_40<input_rgb_image_pyramid<pyramid_down<3>>>>>>>;

#endif //NET_DEFINITION_H
