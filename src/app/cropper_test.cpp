#include "cropper_test.h"

#include <iostream>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_transforms.h>

using namespace std;
using namespace dlib;

void test_cropper(char* fileArg)
{
    try
    {
        std::vector<matrix<rgb_pixel>>      imgs;
        std::vector<std::vector<mmod_rect>> boxes;

        load_image_dataset(imgs, boxes, fileArg);

        random_cropper cropper;

        cropper.set_chip_dims(CHIP_SIZE, CHIP_SIZE);
        cropper.set_min_object_size(MIN_SIZE);
        cropper.set_max_object_size(MAX_SIZE);
        cropper.set_randomly_flip(RANDOM_FLIP);
        cropper.set_max_rotation_degrees(RANDOM_ROTATION_ANGLE);

        std::vector<matrix<rgb_pixel>>      batchImgs;
        std::vector<std::vector<mmod_rect>> batchBoxes;

        cout << "number of images " << imgs.size() << endl;
        cout << "number of boxes " << boxes.size() << endl;
        cout << "batch size " << BATCH_SIZE << endl;

        cropper(BATCH_SIZE, imgs, boxes, batchImgs, batchBoxes);

        cout << "batch img size: " << batchImgs.size() << endl;

        image_window win;
        for (size_t i = 0; i < batchImgs.size(); ++i)
        {
            cout << "Img num: " << i << endl;
            win.clear_overlay();
            win.set_image(batchImgs[i]);
            for (auto b : batchBoxes[i])
            {
                // Note that mmod_rect has an ignore field.  If an object was labeled
                // ignore in boxes then it will still be labeled as ignore in
                // crop_boxes.  Moreover, objects that are not well contained within
                // the crop are also set to ignore.
                if (b.ignore)
                    win.add_overlay(b.rect, rgb_pixel(255,255,0)); // draw ignored boxes as orange
                else
                    win.add_overlay(b.rect, rgb_pixel(255,0,0));   // draw other boxes as red
            }
            cout << "Hit enter to view the next random crop.";
            cin.get();
        }
    }
    catch (std::exception& e)
    {
        cout << "Exception occured!" << endl;
        cout << e.what() << endl;
    }

}
