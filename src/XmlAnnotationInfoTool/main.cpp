#include <iostream>
#include "pugi_xml/pugixml.hpp"

#define NOT_FOUND -555

using namespace pugi;
using namespace std;

struct label_box
{
    int height;
    int width;

    label_box(int width, int height)
    {
        this->width = width;
        this->height = height;
    }

    int area()
    {
        return width * height;
    }

};


int main(int argc, char** argv) {

    if (argc < 2){
        std::cout << "Please pass xml file containing data annotations." << std::endl;
        return 1;
    }

    char* xmlFile = argv[1];

    xml_document document;
    xml_parse_result result = document.load_file(xmlFile);

    cout << "Load result: " << result.description() <<  endl;

    xml_node root = document.child("dataset");
    xml_node images = root.child("images");


    int smallestWidth = 9999;
    int smallestHeight = 9999;
    int biggestWidth = -1;
    int biggestHeight = -1;
    int boxesWrong = 0;

    label_box smallestBox(9999, 9999);
    label_box biggestBox(0,0);


    for (xml_node image : images.children("image"))
    {
        for (xml_node box : image.children("box"))
        {
            int width = box.attribute("width").as_int(NOT_FOUND);
            int height = box.attribute("height").as_int(NOT_FOUND);

            if (width == NOT_FOUND || height == NOT_FOUND)
            {
                ++boxesWrong;
                continue;
            }

            label_box currentBox(width, height);

            if (currentBox.area() < smallestBox.area())
                smallestBox = currentBox;

            if (currentBox.area() > biggestBox.area())
                biggestBox = currentBox;

            if (width < smallestWidth)
                smallestWidth = width;

            if (width > biggestWidth)
                biggestWidth = width;

            if (height < smallestHeight)
                smallestHeight = height;

            if (height > biggestHeight)
                biggestHeight = height;
        }
    }


    cout << boxesWrong << " boxes were ignored." << endl << endl;

    cout << "Smallest width:    " << smallestWidth << endl;
    cout << "Biggest width:     " << biggestWidth << endl;
    cout << "Smallest height:   " << smallestHeight << endl;
    cout << "Biggest height:    " << biggestHeight << endl << endl;

    cout << "Smallest found box [W x H]: [" << smallestBox.width << " x " << smallestBox.height << "]" << endl;
    cout << "Biggest found box [W x H]: [" << biggestBox.width << " x " << biggestBox.height << "]" << endl << endl;

    cout << "Smallest possible box [W x H]: [" << smallestWidth << " x " << smallestHeight << "]" << endl;
    cout << "Biggest possible box [W x H]: [" << biggestHeight << " x " << biggestHeight << "]" << endl;




    return 0;
}