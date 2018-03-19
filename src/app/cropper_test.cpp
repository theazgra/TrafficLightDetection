#include "cropper_test.h"

using namespace std;
using namespace dlib;

void test_cropper(const std::string xmlFile, bool display, bool displayOnly)
{
    try
    {
        cout << endl << "Loading images..." << endl;
        std::vector<matrix<rgb_pixel>>      imgs;
        std::vector<std::vector<mmod_rect>> boxes;

        load_image_dataset(imgs, boxes, xmlFile);

        random_cropper cropper;

        //rows then cols
        cropper.set_chip_dims(CHIP_HEIGHT, CHIP_WIDTH);
        cropper.set_min_object_size(MIN_OBJECT_SIZE_L, MIN_OBJECT_SIZE_S);
        cropper.set_max_object_size(MAX_OBJECT_SIZE);

        cropper.set_background_crops_fraction(BACKGROUND_CROP_FRACTION);

        //defaulted to false
        cropper.set_randomly_flip(RANDOM_FLIP);
        cropper.set_max_rotation_degrees(RANDOM_ROTATION_ANGLE);

        std::vector<matrix<rgb_pixel>>      batchImgs;
        std::vector<std::vector<mmod_rect>> batchBoxes;

        cout << "Number of images: " << imgs.size() << endl;
        cout << "Number of labeling boxes: " << boxes.size() << endl;
        cout << "Batch size is set to: " << BATCH_SIZE << endl << endl;

        cropper(BATCH_SIZE, imgs, boxes, batchImgs, batchBoxes);


        if (!displayOnly)
        {
            cout << endl << "Testing boxes..." << endl;
            int numIgnored = 0;
            int numCorrect = 0;
            int numOverlapped = 0;

            for (size_t i = 0; i < batchImgs.size(); ++i)
            {
                numOverlapped += overlapped_boxes_count(batchBoxes[i], test_box_overlap(OVERLAP_IOU, COVERED_THRESHOLD));
                for (auto b : batchBoxes[i])
                {
                    if (b.ignore)
                        ++numIgnored;
                    else
                        ++numCorrect;
                }
            }

            cout << "Number of overlapping boxes: " << numOverlapped << endl;
            cout << "Number of correct boxes: " << numCorrect << endl;
            cout << "Number of ignored boxes: " << numIgnored << endl;
            float percentCorrect = ((float)numCorrect / (float)(numCorrect + numIgnored)) * 100.0f;
            cout << "Percent of boxes correct: " << percentCorrect << endl;

        }

        dlib::rand rnd;

        if (display || displayOnly)
        {
            cout << endl << "Showing images with boxes, greed for good, red for ignored box." << endl;
            image_window win;

            for (size_t i = 0; i < batchImgs.size(); ++i)
            {
                cout << "Img #: " << i << endl;
                win.clear_overlay();

                win.set_image(batchImgs[i]);
                for (auto b : batchBoxes[i])
                {
                    if (b.ignore)
                        win.add_overlay(b.rect, rgb_pixel(255,0,0));
                    else
                        win.add_overlay(b.rect, rgb_pixel(0,255,0));
                }
                cout << "Hit enter to view the next random crop.";
                cin.get();
            }
        }
    }
    catch (std::exception& e)
    {
        cout << "Exception occured!" << endl;
        cout << e.what() << endl;
    }
}
