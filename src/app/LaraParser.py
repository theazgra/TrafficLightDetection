"""@package LaraParser
    Python script to parse Lara ground truth to imglab xml format.
"""

import xml.etree.ElementTree as ET

class Rectangle(object):
    def __init__(self, left, top, right, bottom):
        self.left = int(left)
        self.right = int(right)
        self.top = int(top)
        self.bottom = int(bottom)
        self.width = self.right - self.left
        self.height = self.bottom - self.top

    def to_attribs(self):
        attribs = {}
        attribs["top"] = str(self.top)
        attribs["left"] = str(self.left)
        attribs["width"] = str(self.width)
        attribs["height"] = str(self.height)

        return attribs

class Annotation(object):
    def __init__(self, frame, left, top, right, bottom, label):
       self.frame = int(frame)
       self.rectangle = Rectangle(left, top, right, bottom)
       self.label = label.replace("'", "")
       self.label = self.label.strip()
    
    def to_node(self, imageNode):
        attribs = self.rectangle.to_attribs()
        ignore = self.label == "ambiguous"
        if ignore:
            attribs["ignore"] = str(1)
        boxNode = ET.SubElement(imageNode, "box", attribs)
        if not ignore:
            ET.SubElement(boxNode, "label").text = self.label


def file_name(frame):
        size = 6
        name = "frame_"
        numStr = str(frame)
        
        numStr = "0"*(size - len(numStr)) + numStr
        name = name + numStr + ".jpg"

        return name
        #for i in range(0, size - len(numStr))
        #    numStr = "0" + numStr


def main():
    lara_lines = []
    annotations = []
    frames = {}
    with open("Lara_UrbanSeq1_GroundTruth_GT.txt", "r") as gt:
        for line in gt:
            if line[0:1] == "#":
                continue
            else:
                lineTokens = line.split(" ")
                annotations.append(Annotation(
                    lineTokens[2], 
                    lineTokens[3], 
                    lineTokens[4], 
                    lineTokens[5], 
                    lineTokens[6],
                    lineTokens[10]))
    #print(len(annotations))

    for anot in annotations:
        if anot.frame not in frames:
            frames[anot.frame] = []
            frames[anot.frame].append(anot)

        else:
            frames[anot.frame].append(anot)
    
    #print(len(frames))


    root = ET.Element("dataset")
    ET.SubElement(root, "name").text = "imglab dataset"
    ET.SubElement(root, "comment").text = "Created by Lara parser"
    images = ET.SubElement(root, "images")

    for frame in frames:
        image = ET.SubElement(images, "image", {"file": "lara/" + file_name(frame)})
        for anot in frames[frame]:
            anot.to_node(image)
    

    
    tree = ET.ElementTree(root)
    tree.write("parsed.xml", encoding='utf-8', xml_declaration=True)
    
    
    

if __name__ == '__main__':
    main()