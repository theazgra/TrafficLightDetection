/** \file net_definition.h
 * File contating all network types and its blocks.
 */

#ifndef NET_DEFINITION_H
#define NET_DEFINITION_H

#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>

using namespace dlib;
/// Convolution of size 5 with stride of 2 (downsampling).
template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
/// Convolution of size 5 with stride of 1.
template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;


/// Downsampler, 3x con5d, downsampling 8x
template <typename SUBNET> using downsampler8x  = relu<bn_con<con5d<32, relu<bn_con<con5d<32, relu<bn_con<con5d<16,SUBNET>>>>>>>>>;
/// Downsampler, 3x con5d, downsampling 8x (bn_con replaced with affine for testing purposes)
template <typename SUBNET> using a_downsampler8x  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;

/// Downsampler, 2x con5d, downsampling 4x
template <typename SUBNET> using downsampler4x  = relu<bn_con<con5d<32, relu<bn_con<con5d<16,SUBNET>>>>>>;
/// Downsampler, 2x con5d, downsampling 4x (bn_con replaced with affine for testing purposes)
template <typename SUBNET> using a_downsampler4x  = relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>;


/// CNN block with filter size = 5 and 55 filters.
template <typename SUBNET> using rcon5_55  = relu<bn_con<con5<55,SUBNET>>>;
/// CNN block with filter size = 5 and 55 filters (affine instead of bn_con)
template <typename SUBNET> using a_rcon5_55  = relu<affine<con5<55,SUBNET>>>;
/// CNN block with filter size = 5 and 50 filters.
template <typename SUBNET> using rcon5_50  = relu<bn_con<con5<50,SUBNET>>>;
/// CNN block with filter size = 5 and 50 filters (affine instead of bn_con)
template <typename SUBNET> using a_rcon5_50  = relu<affine<con5<50,SUBNET>>>;
/// CNN block with filter size = 5 and 40 filters.
template <typename SUBNET> using rcon5_40  = relu<bn_con<con5<40,SUBNET>>>;
/// CNN block with filter size = 5 and 40 filters (affine instead of bn_con)
template <typename SUBNET> using a_rcon5_40  = relu<affine<con5<40,SUBNET>>>;
/// CNN block with filter size = 5 and 32 filters.
template <typename SUBNET> using rcon5_32  = relu<bn_con<con5<32,SUBNET>>>;
/// CNN block with filter size = 5 and 32 filters (affine instead of bn_con)
template <typename SUBNET> using a_rcon5_32  = relu<affine<con5<32,SUBNET>>>;

/// Main CNN type
using net_type = loss_mmod<con<1,9,9,1,1,rcon5_55<rcon5_55<rcon5_55<rcon5_40<downsampler8x<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>>;
/// Test CNN type. Uses affine instead of bn_con
using test_net_type = loss_mmod<con<1,9,9,1,1,a_rcon5_55<a_rcon5_55<a_rcon5_55<a_rcon5_40<a_downsampler8x<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>>;

template <long num_filters, typename SUBNET> using con3  = con<num_filters,3,3,1,1,SUBNET>;
template <typename SUBNET> using rcon3_55  = relu<bn_con<con3<55, SUBNET>>>;
template <typename SUBNET> using rcon3_40  = relu<bn_con<con3<40, SUBNET>>>;
template <typename SUBNET> using arcon3_55  = relu<affine<con3<55, SUBNET>>>;
template <typename SUBNET> using arcon3_40  = relu<affine<con3<40, SUBNET>>>;
using state_net_type = loss_mmod<con<1,9,9,1,1,rcon3_55<rcon3_40<input_rgb_image_pyramid<pyramid_down<3>>>>>>;
using state_test_net_type = loss_mmod<con<1,9,9,1,1,arcon3_55<arcon3_40<input_rgb_image_pyramid<pyramid_down<3>>>>>>;

//**********************************ResNet*****************************************
template <
        long num_filters,
        template <typename> class BN,
        long stride,
        typename SUBNET
>
using block  = BN<con<num_filters,3,3,1,1,relu<BN<con<num_filters,3,3,stride,stride,SUBNET>>>>>;

template <
        template <long,template<typename>class,long,typename> class block,
        long num_filters,
        template<typename>class BN,
        typename SUBNET
>
using residual = add_prev1<block<num_filters,BN,1,tag1<SUBNET>>>;

template <
        template <long,template<typename>class,long,typename> class block,
        long num_filters,
        template<typename>class BN,
        typename SUBNET
>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<num_filters,BN,2,tag1<SUBNET>>>>>>;

template <typename SUBNET> using res       = relu<residual<block,8,bn_con,SUBNET>>;
template <typename SUBNET> using ares      = relu<residual<block,8,affine,SUBNET>>;
template <typename SUBNET> using res_down  = relu<residual_down<block,8,bn_con,SUBNET>>;
template <typename SUBNET> using ares_down = relu<residual_down<block,8,affine,SUBNET>>;

using resnet_net_type = loss_mmod<con<1,9,9,1,1, res<res<res<res_down<res<res<res<res<res<res<res_down<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>>>>>>>>;
//using resnet_net_type = loss_multiclass_log<fc<1,avg_pool_everything<res<res<res<res_down<repeat<9,res,res_down<res<input<matrix<unsigned char>>>>>>>>>>>>;


using resnet_test_net_type = 
loss_mmod<con<1,9,9,1,1,avg_pool_everything<ares<ares<ares<ares_down<repeat<9, ares, ares_down<ares<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>>>>>;

#endif //NET_DEFINITION_H
