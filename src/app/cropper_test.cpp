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

        //rows then cols
        cropper.set_chip_dims(CHIP_HEIGHT, CHIP_WIDTH);
        cropper.set_min_object_size(MIN_OBJECT_SIZE_L, MIN_OBJECT_SIZE_S);
        cropper.set_max_object_size(MAX_OBJECT_SIZE);

        cropper.set_background_crops_fraction(0.25);

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

        int num_ignored = 0;
        int num_correct = 0;
        for (size_t i = 0; i < batchImgs.size(); ++i)
        {
            for (auto b : batchBoxes[i])
            {
                if (b.ignore)
                    ++num_ignored;
                else
                    ++num_correct;
            }
        }

        cout << "Number of correct boxes: " << num_correct << endl;
        cout << "Number of ignored boxes: " << num_ignored << endl;

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
                    win.add_overlay(b.rect, rgb_pixel(255,0,0)); // draw ignored boxes as red
                else
                    win.add_overlay(b.rect, rgb_pixel(0,255,0));   // draw other boxes as green
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
