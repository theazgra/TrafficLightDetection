#ifndef NET_DEFINITION_H
#define NET_DEFINITION_H

#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>

using namespace dlib;

template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;
template <typename SUBNET> using downsampler  = relu<bn_con<con5d<32, relu<bn_con<con5d<32, relu<bn_con<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using downsampler2 = relu<bn_con<con5d<32, relu<bn_con<con5d<32, SUBNET>>>>>>;
template <typename SUBNET> using rcon5  = relu<bn_con<con5<55,SUBNET>>>;
using net_type = loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;
//using net_type = loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler2<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;


#endif //NET_DEFINITION_H
